import json
import re
import time
from pathlib import Path

import requests
import serpapi


# ============================================================
# CONFIGURATION
# ============================================================

# Put your SerpApi key here
SERPAPI_KEY = "<API_KEY>"

# File containing your research topics
TOPIC_FILE = "search_topics.txt"

# Main output directory
OUTPUT_FOLDER = "research_papers"

# Number of Google Scholar results per query
PAPERS_PER_QUERY = 20

# Delay between queries
REQUEST_DELAY = 3

# Download available PDF files
DOWNLOAD_PDFS = True

# SerpApi search engine
SEARCH_ENGINE = "google_scholar"


# ============================================================
# READ TOPICS
# ============================================================

def read_queries():

    topic_path = Path(TOPIC_FILE)

    if not topic_path.exists():

        raise FileNotFoundError(
            f"Topic file not found: {TOPIC_FILE}"
        )

    queries = []

    with open(
        topic_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            # Remove surrounding quotes
            if (
                line.startswith('"')
                and line.endswith('"')
            ):

                line = line[1:-1]

            queries.append(line)

    # Remove duplicate queries
    queries = list(
        dict.fromkeys(queries)
    )

    return queries


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(name):

    # Remove characters that are invalid
    # in Windows/Linux filenames
    name = re.sub(
        r'[<>:"/\\|?*]',
        '',
        name
    )

    # Replace multiple spaces
    name = re.sub(
        r'\s+',
        '_',
        name
    )

    # Prevent excessively long filenames
    return name[:150]


# ============================================================
# CREATE SERPAPI CLIENT
# ============================================================

def create_client():

    if not SERPAPI_KEY:

        raise RuntimeError(
            "SERPAPI_KEY is empty. "
            "Put your SerpApi key in the "
            "SERPAPI_KEY variable."
        )

    return serpapi.Client(
        api_key=SERPAPI_KEY
    )


# ============================================================
# SEARCH GOOGLE SCHOLAR
# ============================================================

def search_google_scholar(
    client,
    query
):

    params = {

        "engine": SEARCH_ENGINE,

        "q": query,

        "num": PAPERS_PER_QUERY,

        "hl": "en"
    }

    try:

        results = client.search(
            params
        )

        return results

    except Exception as error:

        print(
            "\nSerpApi request failed."
        )

        print(
            f"Query: {query}"
        )

        print(
            f"Error: {error}"
        )

        return None


# ============================================================
# EXTRACT YEAR
# ============================================================

def extract_year(text):

    if not text:
        return None

    match = re.search(
        r'\b(19|20)\d{2}\b',
        text
    )

    if match:

        return int(
            match.group()
        )

    return None


# ============================================================
# EXTRACT PAPER INFORMATION
# ============================================================

def extract_paper(
    result,
    query,
    rank
):

    title = result.get(
        "title",
        "Unknown Title"
    )

    link = result.get(
        "link",
        ""
    )

    snippet = result.get(
        "snippet",
        ""
    )

    # --------------------------------------------------------
    # Publication information
    # --------------------------------------------------------

    publication_info = (
        result.get(
            "publication_info",
            {}
        )
    )

    publication_summary = (
        publication_info.get(
            "summary",
            ""
        )
    )

    year = extract_year(
        publication_summary
    )


    # --------------------------------------------------------
    # Authors
    # --------------------------------------------------------

    authors = []

    author_data = (
        publication_info.get(
            "authors",
            []
        )
    )

    if isinstance(
        author_data,
        list
    ):

        for author in author_data:

            if isinstance(
                author,
                dict
            ):

                name = author.get(
                    "name"
                )

                if name:
                    authors.append(
                        name
                    )

            elif isinstance(
                author,
                str
            ):

                authors.append(
                    author
                )


    # --------------------------------------------------------
    # Citation count
    # --------------------------------------------------------

    citation_count = 0

    inline_links = (
        result.get(
            "inline_links",
            {}
        )
    )

    cited_by = (
        inline_links.get(
            "cited_by",
            {}
        )
    )

    citation_count = cited_by.get(
        "total",
        0
    )


    # --------------------------------------------------------
    # SerpApi result ID
    # --------------------------------------------------------

    paper_id = result.get(
        "result_id"
    )

    # Fallback if result_id isn't available
    if not paper_id:

        paper_id = (
            title.lower()
            .strip()
        )

        paper_id = re.sub(
            r'\s+',
            ' ',
            paper_id
        )


    # --------------------------------------------------------
    # PDF URL
    # --------------------------------------------------------

    pdf_url = None

    resources = result.get(
        "resources",
        []
    )

    if isinstance(
        resources,
        list
    ):

        for resource in resources:

            if not isinstance(
                resource,
                dict
            ):
                continue

            resource_link = (
                resource.get(
                    "link"
                )
            )

            if not resource_link:
                continue

            file_format = (
                resource.get(
                    "file_format",
                    ""
                )
                .lower()
            )

            if (
                file_format == "pdf"
                or ".pdf" in resource_link.lower()
            ):

                pdf_url = resource_link

                break


    # --------------------------------------------------------
    # Build paper object
    # --------------------------------------------------------

    return {

        "paper_id": str(
            paper_id
        ),

        "title": title,

        "authors": authors,

        "year": year,

        "publication_summary": (
            publication_summary
        ),

        "citation_count": (
            citation_count
        ),

        "google_scholar_url": (
            link
        ),

        "pdf_url": pdf_url,

        "snippet": snippet,

        "found_by_queries": [
            query
        ],

        "best_rank": rank
    }


# ============================================================
# DOWNLOAD PDF
# ============================================================

def download_pdf(
    pdf_url,
    paper_id,
    pdf_folder
):

    if not pdf_url:

        return None

    filename = (
        safe_filename(
            str(paper_id)
        )
        + ".pdf"
    )

    pdf_path = (
        pdf_folder /
        filename
    )


    # Already downloaded
    if pdf_path.exists():

        print(
            "PDF already exists."
        )

        return str(
            pdf_path
        )


    try:

        print(
            "Downloading PDF..."
        )

        response = requests.get(
            pdf_url,
            timeout=60,
            stream=True,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        if response.status_code != 200:

            print(
                "PDF download failed:"
                f" HTTP {response.status_code}"
            )

            return None


        with open(
            pdf_path,
            "wb"
        ) as file:

            for chunk in response.iter_content(
                chunk_size=8192
            ):

                if chunk:

                    file.write(
                        chunk
                    )


        # Make sure we didn't save
        # an empty/broken response
        if pdf_path.stat().st_size < 1000:

            pdf_path.unlink(
                missing_ok=True
            )

            print(
                "Downloaded file is too small."
            )

            return None


        print(
            f"PDF saved: {pdf_path.name}"
        )

        return str(
            pdf_path
        )


    except requests.RequestException as error:

        print(
            f"PDF download failed: {error}"
        )

        return None


# ============================================================
# SAVE PAPER AS TXT
# ============================================================

def save_paper_text(
    paper,
    paper_number,
    paper_folder
):

    title = paper.get(
        "title",
        "Unknown Title"
    )

    filename = (
        f"{paper_number:03d}_"
        f"{safe_filename(title)}.txt"
    )

    file_path = (
        paper_folder /
        filename
    )


    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TITLE\n"
        )

        file.write(
            "=====\n"
        )

        file.write(
            f"{title}\n\n"
        )


        file.write(
            "PAPER ID\n"
        )

        file.write(
            "========\n"
        )

        file.write(
            f"{paper.get('paper_id')}\n\n"
        )


        file.write(
            "AUTHORS\n"
        )

        file.write(
            "=======\n"
        )

        file.write(
            f"{', '.join(paper.get('authors', []))}\n\n"
        )


        file.write(
            "YEAR\n"
        )

        file.write(
            "====\n"
        )

        file.write(
            f"{paper.get('year') or 'Unknown'}\n\n"
        )


        file.write(
            "PUBLICATION INFORMATION\n"
        )

        file.write(
            "=======================\n"
        )

        file.write(
            f"{paper.get('publication_summary', '')}\n\n"
        )


        file.write(
            "CITATION COUNT\n"
        )

        file.write(
            "==============\n"
        )

        file.write(
            f"{paper.get('citation_count', 0)}\n\n"
        )


        file.write(
            "FOUND BY QUERIES\n"
        )

        file.write(
            "================\n"
        )

        for query in paper.get(
            "found_by_queries",
            []
        ):

            file.write(
                f"- {query}\n"
            )

        file.write(
            "\n"
        )


        file.write(
            "GOOGLE SCHOLAR URL\n"
        )

        file.write(
            "==================\n"
        )

        file.write(
            f"{paper.get('google_scholar_url', '')}\n\n"
        )


        file.write(
            "PDF URL\n"
        )

        file.write(
            "=======\n"
        )

        file.write(
            f"{paper.get('pdf_url') or 'Not available'}\n\n"
        )


        file.write(
            "SNIPPET\n"
        )

        file.write(
            "=======\n"
        )

        file.write(
            paper.get(
                "snippet",
                ""
            )
        )


    return str(
        file_path
    )


# ============================================================
# LOAD EXISTING PAPERS
# ============================================================

def load_existing_papers():

    metadata_file = (
        Path(OUTPUT_FOLDER) /
        "papers.json"
    )

    if not metadata_file.exists():

        return {}


    try:

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        papers = {}

        for paper in data.get(
            "papers",
            []
        ):

            paper_id = paper.get(
                "paper_id"
            )

            if paper_id:

                papers[
                    paper_id
                ] = paper


        print(
            f"Loaded {len(papers)} "
            f"existing papers."
        )

        return papers


    except Exception as error:

        print(
            f"Could not load papers.json: "
            f"{error}"
        )

        return {}


# ============================================================
# LOAD PROCESSED QUERIES
# ============================================================

def load_processed_queries():

    path = (
        Path(OUTPUT_FOLDER) /
        "processed_queries.txt"
    )

    if not path.exists():

        return set()


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return {
            line.strip()
            for line in file
            if line.strip()
        }


# ============================================================
# MARK QUERY AS PROCESSED
# ============================================================

def mark_query_processed(
    query
):

    path = (
        Path(OUTPUT_FOLDER) /
        "processed_queries.txt"
    )

    with open(
        path,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            query + "\n"
        )


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    papers,
    queries
):

    root_folder = Path(
        OUTPUT_FOLDER
    )

    metadata_file = (
        root_folder /
        "papers.json"
    )

    temp_file = (
        root_folder /
        "papers.tmp.json"
    )


    data = {

        "total_queries": len(
            queries
        ),

        "papers_per_query": (
            PAPERS_PER_QUERY
        ),

        "unique_papers": len(
            papers
        ),

        "queries": queries,

        "papers": list(
            papers.values()
        )
    }


    # Save to temporary file first
    # to avoid corrupting papers.json
    # if the program crashes.

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


    temp_file.replace(
        metadata_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "GOOGLE SCHOLAR PAPER COLLECTOR"
    )

    print("=" * 70)


    # ========================================================
    # READ TOPICS
    # ========================================================

    queries = read_queries()


    print(
        f"\nLoaded {len(queries)} "
        f"research queries."
    )


    # ========================================================
    # CREATE SERPAPI CLIENT
    # ========================================================

    client = create_client()


    # ========================================================
    # CREATE FOLDERS
    # ========================================================

    root_folder = Path(
        OUTPUT_FOLDER
    )

    paper_folder = (
        root_folder /
        "papers"
    )

    pdf_folder = (
        root_folder /
        "pdfs"
    )


    root_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    paper_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    if DOWNLOAD_PDFS:

        pdf_folder.mkdir(
            parents=True,
            exist_ok=True
        )


    # ========================================================
    # LOAD PREVIOUS STATE
    # ========================================================

    all_papers = (
        load_existing_papers()
    )

    processed_queries = (
        load_processed_queries()
    )


    paper_counter = len(
        all_papers
    )


    # ========================================================
    # PROCESS EVERY QUERY
    # ========================================================

    for query_number, query in enumerate(
        queries,
        start=1
    ):

        print("\n")

        print("=" * 70)

        print(
            f"QUERY {query_number}/"
            f"{len(queries)}"
        )

        print(
            f"Topic: {query}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # SKIP COMPLETED QUERY
        # ----------------------------------------------------

        if query in processed_queries:

            print(
                "Already processed."
            )

            print(
                "Skipping..."
            )

            continue


        # ====================================================
        # SEARCH
        # ====================================================

        results = search_google_scholar(
            client,
            query
        )


        if results is None:

            print(
                "Search failed."
            )

            print(
                "This query will be retried "
                "on the next run."
            )

            continue


        # ====================================================
        # SERPAPI ERROR
        # ====================================================

        if results.get(
            "error"
        ):

            print(
                "\nSerpApi returned an error:"
            )

            print(
                results.get(
                    "error"
                )
            )

            print(
                "\nQuery was NOT marked "
                "as processed."
            )

            continue


        # ====================================================
        # GET SCHOLAR RESULTS
        # ====================================================

        results_list = results.get(
            "organic_results",
            []
        )


        print(
            f"\nGoogle Scholar returned "
            f"{len(results_list)} results."
        )


        new_papers = 0

        duplicate_papers = 0


        # ====================================================
        # PROCESS PAPERS
        # ====================================================

        for rank, result in enumerate(
            results_list[
                :PAPERS_PER_QUERY
            ],
            start=1
        ):

            paper = extract_paper(
                result,
                query,
                rank
            )


            paper_id = paper[
                "paper_id"
            ]


            # ------------------------------------------------
            # DUPLICATE PAPER
            # ------------------------------------------------

            if paper_id in all_papers:

                existing = (
                    all_papers[
                        paper_id
                    ]
                )


                # Add this query to the
                # paper's matching queries
                if query not in existing[
                    "found_by_queries"
                ]:

                    existing[
                        "found_by_queries"
                    ].append(
                        query
                    )


                # Keep the best ranking
                existing[
                    "best_rank"
                ] = min(
                    existing[
                        "best_rank"
                    ],
                    rank
                )


                duplicate_papers += 1

                continue


            # ------------------------------------------------
            # NEW PAPER
            # ------------------------------------------------

            paper_counter += 1

            paper[
                "local_number"
            ] = paper_counter


            print(
                f"\n[{paper_counter:03d}] "
                f"{paper['title']}"
            )


            # ------------------------------------------------
            # DOWNLOAD PDF
            # ------------------------------------------------

            pdf_path = None


            if (
                DOWNLOAD_PDFS
                and paper.get(
                    "pdf_url"
                )
            ):

                pdf_path = download_pdf(
                    paper[
                        "pdf_url"
                    ],
                    paper_id,
                    pdf_folder
                )


            paper[
                "local_pdf"
            ] = pdf_path


            # ------------------------------------------------
            # SAVE TEXT FILE
            # ------------------------------------------------

            text_path = (
                save_paper_text(
                    paper,
                    paper_counter,
                    paper_folder
                )
            )


            paper[
                "local_text"
            ] = text_path


            # ------------------------------------------------
            # ADD PAPER TO DATABASE
            # ------------------------------------------------

            all_papers[
                paper_id
            ] = paper


            new_papers += 1


        # ====================================================
        # SAVE AFTER EVERY QUERY
        # ====================================================

        save_metadata(
            all_papers,
            queries
        )


        # ====================================================
        # MARK QUERY AS COMPLETE
        # ====================================================

        mark_query_processed(
            query
        )

        processed_queries.add(
            query
        )


        # ====================================================
        # QUERY SUMMARY
        # ====================================================

        print(
            "\n--------------------------------"
        )

        print(
            f"New papers: {new_papers}"
        )

        print(
            f"Duplicate papers: "
            f"{duplicate_papers}"
        )

        print(
            f"Total unique papers: "
            f"{len(all_papers)}"
        )

        print(
            "--------------------------------"
        )


        # ====================================================
        # WAIT
        # ====================================================

        if query_number < len(
            queries
        ):

            print(
                f"\nWaiting "
                f"{REQUEST_DELAY} seconds..."
            )

            time.sleep(
                REQUEST_DELAY
            )


    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_metadata(
        all_papers,
        queries
    )


    # ========================================================
    # SAVE QUERIES
    # ========================================================

    query_file = (
        root_folder /
        "queries.txt"
    )


    with open(
        query_file,
        "w",
        encoding="utf-8"
    ) as file:

        for query in queries:

            file.write(
                query + "\n"
            )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "COLLECTION COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nTotal queries:"
        f" {len(queries)}"
    )

    print(
        f"Processed queries:"
        f" {len(processed_queries)}"
    )

    print(
        f"Unique papers:"
        f" {len(all_papers)}"
    )

    print(
        "\nOutput directory:"
    )

    print(
        root_folder.resolve()
    )

    print(
        "\nMetadata:"
    )

    print(
        (
            root_folder /
            "papers.json"
        ).resolve()
    )


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()
