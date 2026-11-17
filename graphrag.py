import os
import json
import time
import urllib3
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Suppress SSL warnings for government portal certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables explicitly
load_dotenv()

# Configuration & Clients with validation check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

print(f"--- Environment Check ---")
print(f"Loaded NEO4J_URI: {NEO4J_URI}")
print(f"Loaded NEO4J_USERNAME: {NEO4J_USERNAME}")
print(f"Loaded NEO4J_DATABASE: {NEO4J_DATABASE}")
print(f"-------------------------")

if not NEO4J_URI or not NEO4J_PASSWORD or not OPENAI_API_KEY:
    print("[ERROR] Missing environment variables in .env file! Please check your .env configuration.")
    exit(1)

BASE_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def get_soup(url: str) -> BeautifulSoup | None:
    try:
        response = requests.get(url, headers=HEADERS, timeout=20, verify=False)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def scrape_schemes(output_file="tn_schemes.json"):
    print(f"Fetching main scheme list from: {BASE_URL}")
    main_soup = get_soup(BASE_URL)
    if not main_soup:
        print("[ERROR] Could not load main scheme directory.")
        return False

    scheme_links = []
    seen_urls = set()

    for a_tag in main_soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        title = a_tag.get_text(" ", strip=True)

        if ("scheme" in href.lower() and "scheme_list.php" not in href.lower()) and len(title) > 3:
            full_url = urljoin(BASE_URL, href)
            if full_url not in seen_urls:
                seen_urls.add(full_url)
                scheme_links.append({"title": title, "url": full_url})

    print(f"Found {len(scheme_links)} scheme pages to scrape.")
    scraped_data = []

    for idx, item in enumerate(scheme_links, start=1):
        print(f"[{idx}/{len(scheme_links)}] Scraping: {item['title']}")
        detail_soup = get_soup(item["url"])
        if not detail_soup:
            continue

        for tag in detail_soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        table_details = {}
        for row in detail_soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if len(cells) >= 2 and cells[0] and cells[1]:
                table_details[cells[0]] = cells[1]

        main_content = (
            detail_soup.find("div", id="content")
            or detail_soup.find("div", class_="content")
            or detail_soup.body
        )
        raw_text = main_content.get_text("\n", strip=True) if main_content else ""

        scraped_data.append(
            {
                "scheme_name": item["title"],
                "url": item["url"],
                "structured_fields": table_details,
                "full_text": raw_text,
            }
        )
        time.sleep(0.5)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(scraped_data)} schemes to {output_file}")
    return True


class SchemeExtraction(BaseModel):
    official_name: str = Field(description="Official name of the scheme")
    summary: str = Field(description="Concise 2-3 sentence summary of the scheme")
    department: str = Field(description="Department or agency implementing the scheme")
    beneficiaries: List[str] = Field(description="Target groups (e.g., Farmers, Women, Students, MSMEs)")
    eligibility_criteria: List[str] = Field(description="Specific eligibility rules, income limits, or age limits")
    benefits: List[str] = Field(description="Financial aid, subsidies, equipment, or services provided")
    how_to_apply: str = Field(description="Application procedure or required documents")


def extract_scheme_entities(scheme_record: dict) -> SchemeExtraction:
    prompt = f"""
    Extract structured knowledge graph entities from the following Tamil Nadu Government Scheme text.
    Be accurate and preserve specific numbers, subsidy percentages, income limits, and conditions.

    Title: {scheme_record['scheme_name']}
    Table Fields: {json.dumps(scheme_record['structured_fields'], ensure_ascii=False)}
    Page Text:
    {scheme_record['full_text'][:6000]}
    """

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a government policy data extractor."},
            {"role": "user", "content": prompt},
        ],
        response_format=SchemeExtraction,
        temperature=0,
    )
    return response.choices[0].message.parsed


def build_graph_and_vector_store(json_file="tn_schemes.json"):
    if not os.path.exists(json_file):
        print(f"Scraped data file '{json_file}' not found. Running scraper first...")
        success = scrape_schemes(json_file)
        if not success:
            return None, None

    with open(json_file, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Scheme) REQUIRE s.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Beneficiary) REQUIRE b.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Department) REQUIRE d.name IS UNIQUE")

    vector_docs = []

    for idx, item in enumerate(schemes, start=1):
        print(f"[{idx}/{len(schemes)}] Processing KG for: {item['scheme_name']}")
        extracted = extract_scheme_entities(item)

        cypher_upsert = """
        MERGE (s:Scheme {name: $official_name})
        SET s.url = $url,
            s.summary = $summary,
            s.how_to_apply = $how_to_apply,
            s.full_text = $full_text

        MERGE (d:Department {name: $department})
        MERGE (s)-[:MANAGED_BY]->(d)

        WITH s
        UNWIND $beneficiaries AS ben_name
        MERGE (b:Beneficiary {name: toLower(trim(ben_name))})
        MERGE (s)-[:TARGETS_BENEFICIARY]->(b)

        WITH s
        UNWIND $eligibility AS elig_text
        MERGE (e:Eligibility {description: elig_text})
        MERGE (s)-[:HAS_ELIGIBILITY]->(e)

        WITH s
        UNWIND $benefits AS benefit_text
        MERGE (ben:Benefit {description: benefit_text})
        MERGE (s)-[:PROVIDES_BENEFIT]->(ben)
        """

        graph.query(
            cypher_upsert,
            params={
                "official_name": extracted.official_name,
                "url": item["url"],
                "summary": extracted.summary,
                "how_to_apply": extracted.how_to_apply,
                "full_text": item["full_text"][:8000],
                "department": extracted.department or "Government of Tamil Nadu",
                "beneficiaries": extracted.beneficiaries or ["General Public"],
                "eligibility": extracted.eligibility_criteria or ["Refer to scheme guidelines"],
                "benefits": extracted.benefits or ["Refer to scheme guidelines"],
            },
        )

        rich_text = (
            f"Scheme: {extracted.official_name}\n"
            f"Summary: {extracted.summary}\n"
            f"Beneficiaries: {', '.join(extracted.beneficiaries)}\n"
            f"Eligibility: {'; '.join(extracted.eligibility_criteria)}\n"
            f"Benefits: {'; '.join(extracted.benefits)}\n"
            f"How to Apply: {extracted.how_to_apply}\n"
            f"URL: {item['url']}"
        )
        vector_docs.append(
            Document(
                page_content=rich_text,
                metadata={"scheme_name": extracted.official_name, "url": item["url"]},
            )
        )

    print("Building/Updating Neo4j Vector Index...")
    vector_store = Neo4jVector.from_documents(
        vector_docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
        index_name="scheme_vector_index",
        node_label="SchemeChunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )

    graph.refresh_schema()
    print("Knowledge Graph & Vector Store successfully initialized!")
    return graph, vector_store


def run_chatbot(graph, vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    cypher_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        validate_cypher=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        verbose=False,
    )

    final_prompt = ChatPromptTemplate.from_template(
        """You are an official assistant for Tamil Nadu Government Schemes.
Answer the user's question accurately using ONLY the retrieved Knowledge Graph and Semantic Search context below.
If a scheme URL is available in the context, include it as a reference link.
If the information is not present in the context, clearly state that you don't have details for that scheme.

User Question: {question}

--- Structured Knowledge Graph Answer ---
{graph_context}

--- Semantic Document Matches ---
{vector_context}

Helpful, well-structured response:"""
    )

    synthesizer = final_prompt | llm | StrOutputParser()

    print("\n" + "=" * 50)
    print(" Tamil Nadu Schemes GraphRAG Chatbot is Ready! ")
    print(" Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_q:
            continue

        try:
            graph_res = cypher_chain.invoke({"query": user_q})
            graph_answer = graph_res.get("result", "No direct graph match.")
        except Exception as e:
            graph_answer = f"Graph query skipped ({e})"

        docs = vector_store.similarity_search(user_q, k=3)
        vector_context = "\n\n".join([d.page_content for d in docs])

        answer = synthesizer.invoke(
            {
                "question": user_q,
                "graph_context": graph_answer,
                "vector_context": vector_context,
            }
        )

        print(f"\nBot:\n{answer}\n" + "-" * 50)


if __name__ == "__main__":
    print("Starting TN Schemes GraphRAG Pipeline...")
    
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        vector_store = Neo4jVector.from_existing_index(
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            index_name="scheme_vector_index",
        )
        print("Existing Neo4j graph and vector index found. Skipping scrape & build.")
    except Exception as e:
        print(f"Connecting or finding index failed ({e}). Running scrape and building graph...")
        graph, vector_store = build_graph_and_vector_store()

    if graph and vector_store:
        run_chatbot(graph, vector_store)
    else:
        print("[ERROR] Failed to initialize pipeline.")