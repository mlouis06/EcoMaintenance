import base64
import os
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pymupdf
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi


# ============================================================
# basic setup
# keep all the paths in one place so windows doesnt get weird
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"

MANUAL_DIR = BASE_DIR / "manuals"

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_MANUAL_DIR = UPLOAD_DIR / "manuals"

DRAWING_DIR = UPLOAD_DIR / "drawings"


for folder in [
    STATIC_DIR,
    MANUAL_DIR,
    UPLOAD_MANUAL_DIR,
    DRAWING_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


load_dotenv(
    BASE_DIR / ".env"
)


TAVILY_API_KEY = os.getenv(
    "TAVILY_API_KEY"
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)


# small model for turning the evidence into the final clean answer
REFINEMENT_MODEL = os.getenv(
    "OPENAI_REFINEMENT_MODEL",
    "gpt-5.6-luna",
)


# drawings/images are harder so im keeping a stronger model for those
DRAWING_MODEL = os.getenv(
    "OPENAI_DRAWING_MODEL",
    "gpt-5.6-terra",
)


MAX_UPLOAD_BYTES = (
    30 * 1024 * 1024
)

MAX_DRAWING_FILES_PER_QUESTION = 3

MAX_DRAWING_PAGES_PER_FILE = 2

LOCAL_BM25_THRESHOLD = 2.5


# scanned manuals can use local tesseract ocr
# this checks the normal windows install location automatically
DEFAULT_TESSDATA = Path(
    r"C:\Program Files\Tesseract-OCR\tessdata"
)


if (
    not os.getenv("TESSDATA_PREFIX")
    and DEFAULT_TESSDATA.exists()
):
    os.environ[
        "TESSDATA_PREFIX"
    ] = str(
        DEFAULT_TESSDATA
    )


# ============================================================
# fastapi
# ============================================================

app = FastAPI(
    title="EcoMaintenance",
    version="0.5.0",
    description=(
        "Grounded multimodal industrial "
        "maintenance assistant"
    ),
)


app.mount(
    "/static",
    StaticFiles(
        directory=str(
            STATIC_DIR
        )
    ),
    name="static",
)


# ============================================================
# request models
# ============================================================

class MaintenanceQuestion(
    BaseModel
):

    question: str

    # keeping these so older frontend tests still work
    drawing_id: Optional[str] = None

    drawing_page: Optional[int] = None


    # the real frontend can attach more than one visual
    drawing_ids: list[str] = Field(
        default_factory=list
    )


class DrawingQuestion(
    BaseModel
):

    question: str = (
        "Explain the important parts of this "
        "drawing for a maintenance technician."
    )

    page: Optional[int] = None


# ============================================================
# search words
# these are just for retrieval.
# ai makes the actual final answer later.
# ============================================================

STOP_WORDS = {

    "the",
    "is",
    "in",
    "and",
    "to",
    "of",
    "a",
    "for",
    "on",
    "with",
    "as",
    "what",
    "when",
    "where",
    "why",
    "how",
    "by",
    "an",
    "at",
    "from",
    "that",
    "this",
    "it",
    "be",
    "are",
    "or",
    "if",
    "we",
    "you",
    "all",
    "your",
    "our",
    "their",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
    "which",
    "but",
    "not",
    "can",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "my",
    "mine",
    "yours",
    "ours",
    "theirs",
    "its",
    "these",
    "those",
    "then",
    "than",
    "so",
    "such",
    "some",
    "any",
    "each",
    "every",
    "other",
    "another",
    "more",
    "most",
    "less",
    "least",
    "make",
    "makes",
    "making",
    "industrial",
    "please",
    "help",
    "tell",
    "me",
    "about",
    "could",
    "would",
    "should",
    "there",
    "here",
}


EQUIPMENT_TERMS = {

    "motor",
    "pump",
    "bearing",
    "gearbox",
    "conveyor",
    "sensor",
    "valve",
    "compressor",
    "fan",
    "drive",
    "inverter",
    "vfd",
    "plc",
    "relay",
    "breaker",
    "transformer",
    "generator",
    "machine",
    "equipment",
    "actuator",
    "cylinder",
    "servo",
    "encoder",
    "contactor",
    "switch",
    "fuse",
    "shaft",
    "coupling",
    "belt",
    "gear",
    "solenoid",
    "controller",
    "starter",
}


MAINTENANCE_TERMS = (
    EQUIPMENT_TERMS
    | {

        "hydraulic",
        "pneumatic",
        "electrical",
        "mechanical",

        "overheating",
        "overheat",

        "vibration",
        "vibrating",

        "noise",
        "noisy",
        "clicking",
        "rattling",
        "buzzing",
        "humming",

        "leak",
        "leaking",

        "pressure",
        "temperature",

        "fault",
        "error",
        "alarm",

        "failure",
        "failed",

        "maintenance",
        "repair",

        "voltage",
        "current",

        "lubrication",

        "wiring",

        "cooling",

        "alignment",
        "misalignment",

        "trip",
        "tripping",

        "stuck",
        "jammed",

        "worn",
        "wear",

        "hot",
        "heat",

        "start",
        "starting",

        "stopped",
        "stopping",

        "power",

        "inspection",
    }
)


SYNONYMS = {

    "overheating": [
        "overheat",
        "temperature",
        "hot",
        "heat",
        "cooling",
        "ventilation",
        "airflow",
    ],

    "overheat": [
        "overheating",
        "temperature",
        "hot",
        "heat",
        "cooling",
        "ventilation",
        "airflow",
    ],

    "noise": [
        "sound",
        "noisy",
        "rattle",
        "rattling",
        "click",
        "clicking",
        "squeal",
        "whine",
        "hum",
        "buzz",
    ],

    "clicking": [
        "click",
        "noise",
        "sound",
        "rattle",
        "rattling",
    ],

    "rattling": [
        "rattle",
        "noise",
        "sound",
        "clicking",
    ],

    "buzzing": [
        "buzz",
        "hum",
        "humming",
        "noise",
        "sound",
    ],

    "humming": [
        "hum",
        "buzz",
        "buzzing",
        "noise",
        "sound",
    ],

    "vibration": [
        "vibrating",
        "vibrate",
        "imbalance",
        "misalignment",
    ],

    "vibrating": [
        "vibration",
        "vibrate",
        "imbalance",
        "misalignment",
    ],

    "leak": [
        "leaking",
        "leakage",
        "seal",
    ],

    "leaking": [
        "leak",
        "leakage",
        "seal",
    ],

    "trip": [
        "tripping",
        "overload",
        "breaker",
        "fault",
    ],

    "tripping": [
        "trip",
        "overload",
        "breaker",
        "fault",
    ],

    "start": [
        "starting",
        "startup",
        "fails",
        "failure",
    ],
}


# if our manuals dont answer it, these are the websites
# EcoMaintenance is allowed to look at
TRUSTED_DOMAINS = [

    "abb.com",
    "library.e.abb.com",

    "siemens.com",
    "support.industry.siemens.com",

    "rockwellautomation.com",
    "literature.rockwellautomation.com",

    "se.com",
    "schneider-electric.com",

    "eaton.com",

    "emerson.com",

    "ge.com",

    "mitsubishielectric.com",

    "fanucamerica.com",

    "yaskawa.com",

    "omron.com",

    "boschrexroth.com",

    "danfoss.com",

    "weidmuller.com",
]


# ============================================================
# little helpers
# ============================================================

def tokenize(
    text: str
) -> list[str]:

    return [

        word

        for word in re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

        if word not in STOP_WORDS
    ]


def expand_tokens(
    tokens: list[str]
) -> list[str]:

    expanded = list(
        tokens
    )


    for word in tokens:

        expanded.extend(
            SYNONYMS.get(
                word,
                [],
            )
        )


    return expanded


def is_maintenance_question(
    question: str
) -> bool:

    words = set(
        tokenize(
            question
        )
    )


    return bool(
        words
        & MAINTENANCE_TERMS
    )


def safe_filename(
    filename: str
) -> str:

    filename = Path(
        filename
    ).name


    filename = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        filename,
    )


    return (
        filename[:150]
        or "upload"
    )


def friendly_filename(
    path: Path
) -> str:

    # uploaded files start with an id.
    # nobody needs to see that ugly thing in the app.

    parts = path.name.split(
        "_",
        1,
    )


    if (
        len(parts) == 2
        and re.fullmatch(
            r"[0-9a-f]{12}",
            parts[0],
        )
    ):

        return parts[1]


    return path.name


def document_id_from_path(
    path: Path
) -> str:

    parts = path.name.split(
        "_",
        1,
    )


    if (
        len(parts) == 2
        and re.fullmatch(
            r"[0-9a-f]{12}",
            parts[0],
        )
    ):

        return parts[0]


    return (
        f"builtin-{path.stem}"
    )


def clean_text(
    text: str
) -> str:

    text = re.sub(
        r"\s+",
        " ",
        text or " ",
    )


    return text.strip()


def make_excerpt(
    text: str,
    question: str,
    limit: int = 700,
) -> str:

    text = clean_text(
        text
    )


    if len(text) <= limit:

        return text


    words = expand_tokens(
        tokenize(
            question
        )
    )


    lowered = (
        text.lower()
    )


    positions = [

        lowered.find(
            word.lower()
        )

        for word in words

        if lowered.find(
            word.lower()
        ) >= 0
    ]


    if not positions:

        return (
            text[:limit].rstrip()
            + "…"
        )


    center = min(
        positions
    )


    start = max(
        0,
        center - limit // 3,
    )


    end = min(
        len(text),
        start + limit,
    )


    excerpt = text[
        start:end
    ].strip()


    if start > 0:

        excerpt = (
            "…"
            + excerpt
        )


    if end < len(text):

        excerpt += "…"


    return excerpt


def local_ocr_configured() -> bool:

    tessdata = os.getenv(
        "TESSDATA_PREFIX"
    )


    return bool(

        tessdata

        and Path(
            tessdata
        ).exists()
    )


# ============================================================
# manual indexing
# every manual goes into one shared searchable knowledge base
# ============================================================

manual_chunks: list[dict] = []

manual_bm25: Optional[
    BM25Okapi
] = None


def manual_file_paths() -> list[Path]:

    files: list[Path] = []


    for folder in [

        MANUAL_DIR,

        UPLOAD_MANUAL_DIR,
    ]:

        for suffix in [

            "*.pdf",

            "*.txt",

            "*.md",
        ]:

            files.extend(
                folder.glob(
                    suffix
                )
            )


    return sorted(
        set(
            files
        )
    )


def _ocr_page_if_needed(
    page
) -> str:

    # normal pdf text extraction is way faster.
    # only use ocr when the page basically has no text.

    if not local_ocr_configured():

        return ""


    try:

        text_page = (
            page.get_textpage_ocr(
                language="eng",
                dpi=150,
                full=True,
            )
        )


        return clean_text(

            page.get_text(
                "text",
                textpage=text_page,
            )
        )


    except Exception as exc:

        print(
            "OCR skipped:",
            type(exc).__name__,
            exc,
        )


        return ""


def extract_manual_chunks(
    path: Path
) -> list[dict]:

    chunks: list[dict] = []


    suffix = (
        path.suffix.lower()
    )


    try:

        # -----------------------------
        # PDF manuals
        # -----------------------------

        if suffix == ".pdf":

            doc = pymupdf.open(
                str(
                    path
                )
            )


            for page_index in range(
                doc.page_count
            ):

                page = (
                    doc.load_page(
                        page_index
                    )
                )


                text = clean_text(

                    page.get_text(
                        "text"
                    )
                )


                ocr_used = False


                # scanned/image-only page
                if len(text) < 40:

                    ocr_text = (
                        _ocr_page_if_needed(
                            page
                        )
                    )


                    if ocr_text:

                        text = (
                            ocr_text
                        )

                        ocr_used = True


                # still nothing useful
                if len(text) < 40:

                    continue


                chunks.append({

                    "document_id":
                        document_id_from_path(
                            path
                        ),

                    "filename":
                        friendly_filename(
                            path
                        ),

                    "page":
                        page_index + 1,

                    "text":
                        text,

                    "ocr_used":
                        ocr_used,
                })


            doc.close()


        # -----------------------------
        # text / markdown manuals
        # -----------------------------

        elif suffix in {
            ".txt",
            ".md",
        }:

            text = clean_text(

                path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            )


            if text:

                # split giant text files so one file
                # doesnt completely dominate retrieval

                block_size = 5000


                for i in range(
                    0,
                    len(text),
                    block_size,
                ):

                    block = text[
                        i:
                        i + block_size
                    ]


                    chunks.append({

                        "document_id":
                            document_id_from_path(
                                path
                            ),

                        "filename":
                            friendly_filename(
                                path
                            ),

                        "page":
                            (
                                i
                                // block_size
                            )
                            + 1,

                        "text":
                            block,

                        "ocr_used":
                            False,
                    })


    except Exception as exc:

        print(
            f"Could not index "
            f"{path.name}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


    return chunks


def rebuild_manual_index() -> int:

    global manual_chunks

    global manual_bm25


    chunks: list[dict] = []


    for path in manual_file_paths():

        chunks.extend(

            extract_manual_chunks(
                path
            )
        )


    manual_chunks = chunks


    tokenized_corpus = [

        tokenize(
            chunk["text"]
        )

        for chunk in manual_chunks
    ]


    tokenized_corpus = [

        tokens
        if tokens
        else ["empty"]

        for tokens in tokenized_corpus
    ]


    manual_bm25 = (

        BM25Okapi(
            tokenized_corpus
        )

        if tokenized_corpus

        else None
    )


    return len(
        manual_chunks
    )


def search_manuals(
    question: str,
    top_k: int = 6,
    max_per_document: int = 2,
) -> list[dict]:

    if (
        not manual_bm25
        or not manual_chunks
    ):

        return []


    query_tokens = tokenize(
        question
    )


    expanded_query = (
        expand_tokens(
            query_tokens
        )
    )


    if not expanded_query:

        return []


    scores = (
        manual_bm25.get_scores(
            expanded_query
        )
    )


    ranked_indexes = sorted(

        range(
            len(scores)
        ),

        key=lambda i:
            float(
                scores[i]
            ),

        reverse=True,
    )


    equipment_in_query = (

        set(
            query_tokens
        )

        & EQUIPMENT_TERMS
    )


    issue_terms = [

        word

        for word in query_tokens

        if word
        not in EQUIPMENT_TERMS
    ]


    expanded_issue_terms = set(

        expand_tokens(
            issue_terms
        )
    )


    # dont let one manual take every result.
    # if we uploaded 3 manuals i actually want to use all 3.

    per_document = defaultdict(
        int
    )


    results: list[dict] = []


    for index in ranked_indexes:

        chunk = (
            manual_chunks[
                index
            ]
        )


        document_id = (
            chunk[
                "document_id"
            ]
        )


        if (
            per_document[
                document_id
            ]
            >= max_per_document
        ):

            continue


        text_lower = (
            chunk["text"].lower()
        )


        token_counts = Counter(

            tokenize(
                text_lower
            )
        )


        equipment_occurrences = sum(

            token_counts[word]

            for word
            in equipment_in_query
        )


        issue_occurrences = sum(

            token_counts[word]

            for word
            in expanded_issue_terms
        )


        unique_issue_matches = sum(

            1

            for word
            in expanded_issue_terms

            if token_counts[word] > 0
        )


        unique_query_matches = sum(

            1

            for word
            in set(
                query_tokens
            )

            if token_counts[word] > 0
        )


        results.append({

            **chunk,

            "score":
                round(
                    float(
                        scores[
                            index
                        ]
                    ),
                    4,
                ),

            "equipment_occurrences":
                equipment_occurrences,

            "issue_occurrences":
                issue_occurrences,

            "unique_issue_matches":
                unique_issue_matches,

            "unique_query_matches":
                unique_query_matches,
        })


        per_document[
            document_id
        ] += 1


        if (
            len(results)
            >= top_k
        ):

            break


    return results


def local_result_is_strong(
    question: str,
    result: Optional[dict],
) -> bool:

    if not result:

        return False


    query_tokens = tokenize(
        question
    )


    equipment_in_query = (

        set(
            query_tokens
        )

        & EQUIPMENT_TERMS
    )


    issue_terms = [

        word

        for word
        in query_tokens

        if word
        not in EQUIPMENT_TERMS
    ]


    # im keeping this conservative.
    # id rather check manufacturer sources than confidently answer with garbage.

    if issue_terms:

        equipment_ok = (

            not equipment_in_query

            or result[
                "equipment_occurrences"
            ] >= 1
        )


        return (

            result["score"]
            >= LOCAL_BM25_THRESHOLD

            and result[
                "issue_occurrences"
            ] >= 2

            and result[
                "unique_issue_matches"
            ] >= 1

            and equipment_ok
        )


    return (

        result["score"]
        >= 4.0

        and result[
            "unique_query_matches"
        ] >= 2
    )


def get_manual_path(
    document_id: str
) -> Path:

    for path in manual_file_paths():

        if (
            document_id_from_path(
                path
            )
            == document_id
        ):

            return path


    raise HTTPException(
        status_code=404,
        detail="Manual not found",
    )


def build_local_evidence(
    results: list[dict],
    question: str,
    max_results: int = 4,
    start_number: int = 1,
) -> tuple[
    str,
    list[dict],
]:

    evidence_parts: list[str] = []

    sources: list[dict] = []


    for offset, result in enumerate(
        results[
            :max_results
        ]
    ):

        number = (
            start_number
            + offset
        )


        file_url = (

            f"/documents/"
            f"{result['document_id']}"
            f"/file"
        )


        if str(
            result["filename"]
        ).lower().endswith(
            ".pdf"
        ):

            file_url += (
                f"#page="
                f"{result['page']}"
            )


        evidence_parts.append(

            f"SOURCE {number}\n"

            f"Type: local manual\n"

            f"File: "
            f"{result['filename']}\n"

            f"Page: "
            f"{result['page']}\n"

            f"Evidence: "
            f"{result['text'][:4500]}"
        )


        sources.append({

            "source_number":
                number,

            "type":
                "local_manual",

            "title":
                result[
                    "filename"
                ],

            "page":
                result[
                    "page"
                ],

            "url":
                file_url,

            "score":
                result[
                    "score"
                ],

            "excerpt":
                make_excerpt(
                    result[
                        "text"
                    ],
                    question,
                ),

            "ocr_used":
                result.get(
                    "ocr_used",
                    False,
                ),
        })


    return (

        "\n\n".join(
            evidence_parts
        ),

        sources,
    )


# build the knowledge base when the app starts
rebuild_manual_index()


# ============================================================
# trusted manufacturer search
# this only happens if our own manuals dont give us enough
# ============================================================

def preferred_manufacturer_domains(
    question: str
) -> list[str]:

    text = question.lower()

    manufacturer_domains = {

        "abb": [
            "abb.com",
            "library.e.abb.com",
        ],

        "siemens": [
            "siemens.com",
            "support.industry.siemens.com",
        ],

        "rockwell": [
            "rockwellautomation.com",
            "literature.rockwellautomation.com",
        ],

        "allen bradley": [
            "rockwellautomation.com",
            "literature.rockwellautomation.com",
        ],

        "schneider": [
            "se.com",
            "schneider-electric.com",
        ],

        "eaton": [
            "eaton.com",
        ],

        "emerson": [
            "emerson.com",
        ],

        "mitsubishi": [
            "mitsubishielectric.com",
        ],

        "yaskawa": [
            "yaskawa.com",
        ],

        "omron": [
            "omron.com",
        ],

        "danfoss": [
            "danfoss.com",
        ],
    }


    for manufacturer, domains in manufacturer_domains.items():

        if manufacturer in text:

            return domains


    return TRUSTED_DOMAINS

def external_search(
    question: str
) -> list[dict]:

    if not TAVILY_API_KEY:

        raise RuntimeError(
            "TAVILY_API_KEY is missing from .env"
        )


    domains = preferred_manufacturer_domains(
        question
    )


    query = (

        f"{question.strip()} "
        f"technical manual specification "
        f"troubleshooting documentation"
    )


    query = query[:390]


    response = requests.post(

        "https://api.tavily.com/search",

        headers={

            "Authorization":
                f"Bearer {TAVILY_API_KEY}",

            "Content-Type":
                "application/json",
        },

        json={

            "query":
                query,

            "topic":
                "general",

            "search_depth":
                "advanced",

            "chunks_per_source":
                3,

            "include_domains":
                domains,

            "max_results":
                4,

            "include_answer":
                False,

            "include_raw_content":
                False,
        },

        timeout=20,
    )


    response.raise_for_status()


    data = response.json()


    results = []


    for item in data.get(
        "results",
        [],
    ):

        content = clean_text(

            item.get(
                "content",
                "",
            )
        )


        if not content:

            continue


        results.append({

            "title":
                item.get(
                    "title"
                )
                or "Manufacturer documentation",

            "url":
                item.get(
                    "url"
                ),

            "content":
                content,

            "score":
                float(
                    item.get(
                        "score",
                        0,
                    )
                    or 0
                ),
        })


    return results


def build_web_evidence(
    results: list[dict],
    question: str,
    max_results: int = 3,
    start_number: int = 1,
) -> tuple[
    str,
    list[dict],
]:

    evidence_parts: list[str] = []

    sources: list[dict] = []


    for offset, result in enumerate(
        results[
            :max_results
        ]
    ):

        number = (
            start_number
            + offset
        )


        evidence_parts.append(

            f"SOURCE {number}\n"

            f"Type: trusted "
            f"manufacturer web result\n"

            f"Title: "
            f"{result['title']}\n"

            f"URL: "
            f"{result['url']}\n"

            f"Evidence: "
            f"{result['content'][:4500]}"
        )


        sources.append({

            "source_number":
                number,

            "type":
                "trusted_web",

            "title":
                result[
                    "title"
                ],

            "page":
                None,

            "url":
                result[
                    "url"
                ],

            "score":
                round(
                    result[
                        "score"
                    ],
                    4,
                ),

            "excerpt":
                make_excerpt(
                    result[
                        "content"
                    ],
                    question,
                ),

            "ocr_used":
                False,
        })


    return (

        "\n\n".join(
            evidence_parts
        ),

        sources,
    )


# ============================================================
# drawing / image vision
# pdf drawings get rendered first so the model can actually see them
# ============================================================

def get_drawing_path(
    document_id: str
) -> Path:

    matches = list(

        DRAWING_DIR.glob(
            f"{document_id}_*"
        )
    )


    if not matches:

        raise HTTPException(
            status_code=404,
            detail=(
                "Drawing or image "
                "not found"
            ),
        )


    return matches[0]


def bytes_to_data_url(
    data: bytes,
    mime_type: str,
) -> str:

    encoded = (
        base64.b64encode(
            data
        ).decode(
            "ascii"
        )
    )


    return (

        f"data:"
        f"{mime_type};"
        f"base64,"
        f"{encoded}"
    )


def drawing_images(
    path: Path,
    requested_page: Optional[int] = None,
) -> list[dict]:

    suffix = (
        path.suffix.lower()
    )


    # normal image
    if suffix in {

        ".png",

        ".jpg",

        ".jpeg",

        ".webp",
    }:

        mime_map = {

            ".png":
                "image/png",

            ".jpg":
                "image/jpeg",

            ".jpeg":
                "image/jpeg",

            ".webp":
                "image/webp",
        }


        return [{

            "page":
                1,

            "data_url":
                bytes_to_data_url(

                    path.read_bytes(),

                    mime_map[
                        suffix
                    ],
                ),
        }]


    # drawings should be pdf for the prototype
    if suffix != ".pdf":

        raise HTTPException(

            status_code=400,

            detail=(
                "Visual support is "
                "PDF, PNG, JPG/JPEG, "
                "or WEBP. Export "
                "DWG/DXF to PDF first."
            ),
        )


    doc = pymupdf.open(
        str(
            path
        )
    )


    if doc.page_count == 0:

        raise HTTPException(

            status_code=400,

            detail=(
                "The drawing PDF "
                "has no pages"
            ),
        )


    if requested_page is not None:

        if (
            requested_page < 1
            or requested_page
            > doc.page_count
        ):

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Page must be "
                    f"between 1 and "
                    f"{doc.page_count}"
                ),
            )


        page_numbers = [

            requested_page - 1
        ]


    else:

        page_numbers = list(

            range(

                min(
                    doc.page_count,
                    MAX_DRAWING_PAGES_PER_FILE,
                )
            )
        )


    images: list[dict] = []


    # bigger render so small labels/wires arent impossible to read
    matrix = pymupdf.Matrix(
        2.0,
        2.0,
    )


    for page_index in page_numbers:

        page = doc.load_page(
            page_index
        )


        pix = page.get_pixmap(
            matrix=matrix,
            alpha=False,
        )


        images.append({

            "page":
                page_index + 1,

            "data_url":
                bytes_to_data_url(

                    pix.tobytes(
                        "png"
                    ),

                    "image/png",
                ),
        })


    doc.close()


    return images


def analyze_visual_bundle(
    paths: list[Path],
    question: str,
) -> dict:

    if not OPENAI_API_KEY:

        raise RuntimeError(
            "OPENAI_API_KEY "
            "is missing from .env"
        )


    content: list[dict] = [{

        "type":
            "input_text",

        "text":
            f"""
You are EcoMaintenance reading industrial electrical drawings,
mechanical drawings, wiring diagrams, equipment photos,
schematics, tables, or other maintenance visuals.

Technician question:
{question}

Read only what is actually visible in the uploaded files.

For every useful observation, cite the visual file using
[Source 1], [Source 2], etc.

Focus on:
- the type of drawing or image
- component tags
- labels
- symbols
- wires and connections
- dimensions
- parts
- visible wear or damage
- mechanical relationships
- the visible path or relationship that matters to the question
- warnings
- ratings
- callouts
- notes
- anything unclear or unreadable

Rules:
- Do not invent a component, wire, value, dimension,
  connection, fault, or condition that is not visible.
- Do not assume equipment is energized or de-energized.
- Do not assume something is safe just from an image.
- Do not recommend bypassing interlocks, guards,
  protective devices, or safety systems.
- If the visuals only partially help, still report what
  they DO show.
- Keep the analysis useful and concise enough to become
  evidence for the final troubleshooting answer.
""".strip(),
    }]


    visual_meta: list[dict] = []


    for (
        source_number,
        path,
    ) in enumerate(
        paths,
        start=1,
    ):

        images = drawing_images(
            path
        )


        content.append({

            "type":
                "input_text",

            "text":
                (
                    f"SOURCE "
                    f"{source_number}: "
                    f"{friendly_filename(path)}"
                ),
        })


        for image in images:

            content.append({

                "type":
                    "input_image",

                "image_url":
                    image[
                        "data_url"
                    ],

                "detail":
                    "high",
            })


        visual_meta.append({

            "source_number":
                source_number,

            "type":
                "uploaded_visual",

            "title":
                friendly_filename(
                    path
                ),

            "page":
                None,

            "pages_analyzed": [

                image[
                    "page"
                ]

                for image in images
            ],

            "url":
                (
                    f"/drawings/"
                    f"{document_id_from_path(path)}"
                    f"/file"
                ),

            "score":
                None,

            "ocr_used":
                False,
        })


    client = OpenAI(
        api_key=OPENAI_API_KEY
    )


    response = (
        client.responses.create(

            model=
                DRAWING_MODEL,

            input=[{

                "role":
                    "user",

                "content":
                    content,
            }],

            max_output_tokens=
                1200,
        )
    )


    analysis = (
        response.output_text
        or ""
    ).strip()


    if not analysis:

        raise RuntimeError(
            "The drawing model "
            "returned an empty response"
        )


    for source in visual_meta:

        source[
            "excerpt"
        ] = make_excerpt(

            analysis,

            question,

            limit=600,
        )


    return {

        "analysis":
            analysis,

        "sources":
            visual_meta,

        "model":
            DRAWING_MODEL,
    }


# ============================================================
# answer refinement
# this turns all the evidence into the clean answer the user sees
# ============================================================

def refine_answer(
    question: str,
    evidence: str,
) -> dict:

    if not OPENAI_API_KEY:

        return {

            "status":
                "not_configured",

            "evidence_status":
                None,

            "answer":
                None,

            "model":
                None,
        }


    client = OpenAI(
        api_key=OPENAI_API_KEY
    )


    try:

        response = (
            client.responses.create(

                model=
                    REFINEMENT_MODEL,

                input=
                    f"""
You are EcoMaintenance, a grounded industrial maintenance assistant.

Technician question:
{question}

Retrieved technical evidence:
{evidence}

Use ONLY the evidence above.

Your first line must be exactly one of these:

EVIDENCE_STATUS: DIRECT
EVIDENCE_STATUS: PARTIAL
EVIDENCE_STATUS: UNRELATED


Definitions:

DIRECT
The supplied evidence clearly addresses the technician's question.

PARTIAL
The supplied evidence contains useful information, possible causes,
checks, procedures, observations, or related technical guidance,
but it does not prove one definite answer.

UNRELATED
The supplied evidence genuinely does not address the question.


Then write the technician-facing answer.


Rules:

- If the evidence is DIRECT or PARTIAL, ALWAYS give the most
  useful grounded answer you can.

- Do not refuse just because the sources are incomplete.

- For PARTIAL evidence, summarize what the sources DO support
  and clearly explain what remains uncertain.

- Never turn a possible cause into a confirmed diagnosis.

- Give practical troubleshooting checks only when the
  evidence supports them.

- Combine useful information from multiple manuals,
  drawings, images, and manufacturer sources.

- Ignore marketing copy, navigation text, contact information,
  and unrelated boilerplate.

- Do not invent specifications, measurements, procedures,
  safety requirements, component values, or equipment details.

- Mention safety precautions when they are supported
  by the evidence.

- Cite technical claims using [Source 1], [Source 2], etc.

- If several sources support the same claim, cite all of them.

- Only use UNRELATED when the retrieved material genuinely
  does not help answer the technician's question.

- Keep the answer clean and easy to scan.

- Start with a short direct answer.

- Use bullets for useful causes, checks, or observations.

- Include a short Safety section only when the evidence
  supports one.

- If the evidence does not confirm one root cause, end with
  one short sentence explaining what observation, measurement,
  manual, drawing, or test would narrow the diagnosis.

- Do not mention AI, models, search levels, retrieval scores,
  confidence scores, or internal processing.
""".strip(),

                max_output_tokens=
                    1100,
            )
        )


        text = (
            response.output_text
            or ""
        ).strip()


        status_match = re.match(

            r"^EVIDENCE_STATUS:\s*"
            r"(DIRECT|PARTIAL|UNRELATED)",

            text,

            flags=
                re.IGNORECASE,
        )


        evidence_status = (

            status_match
            .group(1)
            .lower()

            if status_match

            else "partial"
        )


        answer = re.sub(

            r"^EVIDENCE_STATUS:\s*"
            r"(DIRECT|PARTIAL|UNRELATED)\s*",

            "",

            text,

            flags=
                re.IGNORECASE,
        ).strip()


        return {

            "status":
                "completed",

            "evidence_status":
                evidence_status,

            "answer":
                answer or None,

            "model":
                REFINEMENT_MODEL,
        }


    except Exception as exc:

        print(
            f"AI refinement error: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


        return {

            "status":
                "unavailable",

            "evidence_status":
                None,

            "answer":
                None,

            "model":
                REFINEMENT_MODEL,
        }


# ============================================================
# uploads
# ============================================================

async def save_upload(
    file: UploadFile,
    folder: Path,
    allowed_extensions: set[str],
) -> tuple[
    str,
    Path,
    int,
]:

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "No filename "
                "was provided"
            ),
        )


    original_name = safe_filename(
        file.filename
    )


    suffix = Path(
        original_name
    ).suffix.lower()


    if (
        suffix
        not in allowed_extensions
    ):

        allowed = ", ".join(
            sorted(
                allowed_extensions
            )
        )


        raise HTTPException(

            status_code=400,

            detail=(
                f"Unsupported file type. "
                f"Allowed: {allowed}"
            ),
        )


    document_id = (
        uuid.uuid4()
        .hex[:12]
    )


    path = (

        folder
        / (
            f"{document_id}_"
            f"{original_name}"
        )
    )


    total = 0


    try:

        with path.open(
            "wb"
        ) as output:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )


                if not chunk:

                    break


                total += len(
                    chunk
                )


                if (
                    total
                    > MAX_UPLOAD_BYTES
                ):

                    raise HTTPException(

                        status_code=413,

                        detail=(
                            "File is too large. "
                            "Prototype limit is "
                            "30 MB per file."
                        ),
                    )


                output.write(
                    chunk
                )


    except Exception:

        if path.exists():

            path.unlink()


        raise


    finally:

        await file.close()


    return (
        document_id,
        path,
        total,
    )


async def _save_manual_upload(
    file: UploadFile
) -> dict:

    (
        document_id,
        path,
        size,
    ) = await save_upload(

        file,

        UPLOAD_MANUAL_DIR,

        {
            ".pdf",
            ".txt",
            ".md",
        },
    )


    chunks = (
        extract_manual_chunks(
            path
        )
    )


    ocr_pages = sum(

        1

        for chunk
        in chunks

        if chunk.get(
            "ocr_used"
        )
    )


    warning = None


    if not chunks:

        warning = (

            "The file uploaded, "
            "but no searchable text "
            "was extracted. If it is "
            "scanned, install Tesseract "
            "OCR or upload a searchable PDF."
        )


    return {

        "status":
            "uploaded",

        "document_id":
            document_id,

        "filename":
            friendly_filename(
                path
            ),

        "size_bytes":
            size,

        "searchable_pages_or_chunks":
            len(
                chunks
            ),

        "ocr_pages":
            ocr_pages,

        "warning":
            warning,
    }


async def _save_drawing_upload(
    file: UploadFile
) -> dict:

    (
        document_id,
        path,
        size,
    ) = await save_upload(

        file,

        DRAWING_DIR,

        {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        },
    )


    return {

        "status":
            "uploaded",

        "document_id":
            document_id,

        "filename":
            friendly_filename(
                path
            ),

        "size_bytes":
            size,

        "supported_for_vision":
            True,
    }

def needs_exact_fact_verification(
    question: str
) -> bool:

    text = question.lower()


    exact_terms = {

        "diameter",
        "length",
        "width",
        "height",
        "dimension",
        "dimensions",

        "voltage",
        "current",
        "torque",
        "speed",
        "frequency",
        "power",

        "pin",
        "pins",
        "connector",
        "connectors",

        "parameter",
        "rating",
        "rated",

        "mm",
        "cm",
        "inch",
        "inches",

        "amp",
        "amps",
        "volt",
        "volts",

        "rpm",
        "hz",
        "kw",
        "nm",

        "model",
        "part number",
        "specification",
        "specifications",
    }


    return any(

        term in text

        for term in exact_terms
    )


def verify_exact_facts(
    question: str,
    evidence: str,
    draft_answer: str,
) -> dict:

    if not OPENAI_API_KEY:

        return {

            "status":
                "not_configured",

            "answer":
                draft_answer,
        }


    client = OpenAI(
        api_key=OPENAI_API_KEY
    )


    try:

        response = client.responses.create(

            model=
                REFINEMENT_MODEL,

            input=f"""
You are the final factual verifier for EcoMaintenance.

Technician question:
{question}

Retrieved evidence:
{evidence}

Draft answer:
{draft_answer}


Your job is NOT to create a new answer from general knowledge.

Your job is to audit the draft against the supplied evidence.


STRICT RULES:

1. Use ONLY the supplied evidence.

2. Verify every:
   - number
   - dimension
   - voltage
   - current
   - torque
   - speed
   - frequency
   - pin count
   - connector count
   - parameter
   - model number
   - part number
   - product-specific specification

3. A value is valid only if the evidence clearly associates it
   with the requested model, model family, table row, drawing,
   or specification.

4. Pay special attention to tables.

   Read:
   - the correct ROW
   - the correct COLUMN
   - the correct PRODUCT FAMILY

5. Never copy a value from a neighboring model.

   Example:
   If the question asks about 9C4.3, a value belonging to 9C5,
   9C5.3, 9C3, or another family must NOT be used.

6. A family-level drawing may support a model within that family
   only when the evidence clearly identifies the drawing as
   belonging to that family.

7. If one source is for the exact requested product and another
   source is for a different product family, prefer the exact
   requested product source.

8. If two relevant sources genuinely disagree:
   - say they disagree
   - cite both
   - do not silently choose one

9. Remove unrelated sources from the reasoning.

10. Preserve valid citations such as [Source 1].

11. Do not invent a missing number.

12. If a number cannot be verified, say that the available
    evidence does not establish that value.

13. Return ONLY the corrected technician-facing answer.
Do not write:
VERIFIED
UNVERIFIED
EVIDENCE_STATUS
or any internal notes.
""".strip(),

            max_output_tokens=1000,
        )


        verified_answer = (
            response.output_text
            or ""
        ).strip()


        if not verified_answer:

            return {

                "status":
                    "empty",

                "answer":
                    draft_answer,
            }


        return {

            "status":
                "completed",

            "answer":
                verified_answer,
        }


    except Exception as exc:

        print(
            "Exact fact verification error:",
            type(exc).__name__,
            exc,
        )


        return {

            "status":
                "unavailable",

            "answer":
                draft_answer,
        }

# ============================================================
# routes
# ============================================================

@app.get("/")
def home():

    index_path = (

        STATIC_DIR
        / "index.html"
    )


    if index_path.exists():

        return FileResponse(
            index_path
        )


    return {

        "app":
            "EcoMaintenance",

        "status":
            "Backend is running",

        "message":
            (
                "static/index.html "
                "has not been created yet"
            ),
    }


@app.get("/health")
def health():

    return {

        "app":
            "EcoMaintenance",

        "status":
            "running",

        "manual_files":
            len(
                manual_file_paths()
            ),

        "manual_pages_indexed":
            len(
                manual_chunks
            ),

        "drawings_uploaded":
            len([

                path

                for path
                in DRAWING_DIR.iterdir()

                if path.is_file()
            ]),

        "local_ocr_configured":
            local_ocr_configured(),

        "tavily_configured":
            bool(
                TAVILY_API_KEY
            ),

        "openai_configured":
            bool(
                OPENAI_API_KEY
            ),
    }


@app.get("/documents")
def documents():

    manuals = [

        {

            "document_id":
                document_id_from_path(
                    path
                ),

            "name":
                friendly_filename(
                    path
                ),

            "type":
                "manual",
        }

        for path
        in manual_file_paths()
    ]


    drawings = [

        {

            "document_id":
                document_id_from_path(
                    path
                ),

            "name":
                friendly_filename(
                    path
                ),

            "type":
                "drawing",
        }

        for path
        in sorted(
            DRAWING_DIR.iterdir()
        )

        if path.is_file()
    ]


    return {

        "manuals":
            manuals,

        "drawings":
            drawings,
    }


@app.get(
    "/documents/{document_id}/file"
)
def open_manual(
    document_id: str
):

    path = get_manual_path(
        document_id
    )


    return FileResponse(

        path,

        filename=
            friendly_filename(
                path
            ),
    )


@app.get(
    "/drawings/{document_id}/file"
)
def open_drawing(
    document_id: str
):

    path = get_drawing_path(
        document_id
    )


    return FileResponse(

        path,

        filename=
            friendly_filename(
                path
            ),
    )


# ============================================================
# upload one manual
# keeping this around for swagger / old tests
# ============================================================

@app.post("/upload/manual")
async def upload_manual(
    file: UploadFile = File(...)
):

    result = await (
        _save_manual_upload(
            file
        )
    )


    result[
        "total_manual_pages_or_chunks_indexed"
    ] = rebuild_manual_index()


    return result


# ============================================================
# upload multiple manuals
# this is what the real frontend uses
# ============================================================

@app.post("/upload/manuals")
async def upload_manuals(
    files: list[
        UploadFile
    ] = File(...)
):

    if not files:

        raise HTTPException(

            status_code=400,

            detail=(
                "Choose at least "
                "one manual"
            ),
        )


    results = []


    for file in files:

        results.append(

            await _save_manual_upload(
                file
            )
        )


    total = (
        rebuild_manual_index()
    )


    return {

        "status":
            "uploaded",

        "uploaded_count":
            len(
                results
            ),

        "files":
            results,

        "total_manual_pages_or_chunks_indexed":
            total,
    }


# ============================================================
# upload one drawing/image
# ============================================================

@app.post("/upload/drawing")
async def upload_drawing(
    file: UploadFile = File(...)
):

    return await (
        _save_drawing_upload(
            file
        )
    )


# ============================================================
# upload multiple drawings/images
# ============================================================

@app.post("/upload/drawings")
async def upload_drawings(
    files: list[
        UploadFile
    ] = File(...)
):

    if not files:

        raise HTTPException(

            status_code=400,

            detail=(
                "Choose at least one "
                "drawing or image"
            ),
        )


    results = []


    for file in files:

        results.append(

            await _save_drawing_upload(
                file
            )
        )


    return {

        "status":
            "uploaded",

        "uploaded_count":
            len(
                results
            ),

        "files":
            results,
    }


@app.post("/reindex")
def reindex():

    total = (
        rebuild_manual_index()
    )


    return {

        "status":
            "reindexed",

        "manual_pages_or_chunks_indexed":
            total,
    }


# ============================================================
# test one drawing directly in swagger
# ============================================================

@app.post(
    "/analyze-drawing/{document_id}"
)
def analyze_drawing(
    document_id: str,
    data: DrawingQuestion,
):

    path = get_drawing_path(
        document_id
    )


    try:

        result = (
            analyze_visual_bundle(
                [path],
                data.question,
            )
        )


    except HTTPException:

        raise


    except Exception as exc:

        raise HTTPException(

            status_code=503,

            detail=(
                "Drawing analysis "
                "is unavailable: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


    return {

        "document_id":
            document_id,

        "filename":
            friendly_filename(
                path
            ),

        "analysis":
            result[
                "analysis"
            ],

        "sources":
            result[
                "sources"
            ],
    }


# ============================================================
# main question endpoint
# ============================================================

@app.post("/ask")
def ask_question(
    data: MaintenanceQuestion
):

    question = (
        data.question.strip()
    )


    if not question:

        raise HTTPException(

            status_code=400,

            detail=(
                "Please enter a "
                "maintenance question"
            ),
        )


    # ========================================================
    # figure out which drawings/images the user attached
    # ========================================================

    drawing_ids = []


    # old frontend compatibility
    if data.drawing_id:

        drawing_ids.append(
            data.drawing_id
        )


    # current frontend
    drawing_ids.extend(
        data.drawing_ids
    )


    # remove duplicates but keep the order
    drawing_ids = list(
        dict.fromkeys(
            drawing_ids
        )
    )


    # dont send a ridiculous number of images into one question
    drawing_ids = (
        drawing_ids[
            :MAX_DRAWING_FILES_PER_QUESTION
        ]
    )


    # if its obviously unrelated and theres no visual attached,
    # dont waste search/api calls on it
    if (
        not is_maintenance_question(
            question
        )
        and not drawing_ids
    ):

        return {

            "question":
                question,

            "answer":
                (
                    "This appears to be "
                    "outside EcoMaintenance's "
                    "industrial maintenance scope."
                ),

            "sources":
                [],
        }


    # ========================================================
    # read selected drawings/images
    # ========================================================

    drawing_evidence = ""

    drawing_sources: list[
        dict
    ] = []


    if drawing_ids:

        paths = [

            get_drawing_path(
                document_id
            )

            for document_id
            in drawing_ids
        ]


        try:

            visual_result = (
                analyze_visual_bundle(
                    paths,
                    question,
                )
            )


            drawing_evidence = (

                "VISUAL EVIDENCE\n"

                f"{visual_result['analysis']}"
            )


            drawing_sources = (
                visual_result[
                    "sources"
                ]
            )


        except HTTPException:

            raise


        except Exception as exc:

            raise HTTPException(

                status_code=503,

                detail=(
                    "Could not analyze "
                    "the selected drawing/image: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )


    # ========================================================
    # search ALL uploaded and built in manuals together
    # ========================================================

    local_results = search_manuals(

        question,

        top_k=6,

        max_per_document=2,
    )


    local_strong = (
        local_result_is_strong(

            question,

            (
                local_results[0]

                if local_results

                else None
            ),
        )
    )


    sources: list[
        dict
    ] = list(
        drawing_sources
    )


    evidence_parts: list[
        str
    ] = []


    if drawing_evidence:

        evidence_parts.append(
            drawing_evidence
        )


    # ========================================================
    # strong local manual result
    # ========================================================

    if local_results:

        (
            local_evidence,
            local_sources,
        ) = build_local_evidence(

            local_results,

            question,

            max_results=4,

            start_number=
                len(
                    sources
                )
                + 1,
        )


        # if its a strong match we can start with our own manuals
        if local_strong:

            evidence_parts.append(
                local_evidence
            )


            sources.extend(
                local_sources
            )


    # ========================================================
    # weak local match
    # quietly check trusted manufacturers too
    # ========================================================

    if not local_strong:

        try:

            web_results = (
                external_search(
                    question
                )
            )


        except Exception as exc:

            print(
                "External search error:",
                type(exc).__name__,
                exc,
            )


            web_results = []


        if web_results:

            (
                web_evidence,
                web_sources,
            ) = build_web_evidence(

                web_results,

                question,

                max_results=3,

                start_number=
                    len(
                        sources
                    )
                    + 1,
            )


            evidence_parts.append(
                web_evidence
            )


            sources.extend(
                web_sources
            )


        elif local_results:

            # tavily can fail or find nothing.
            # dont throw away the manuals we DID find.

            (
                local_evidence,
                local_sources,
            ) = build_local_evidence(

                local_results,

                question,

                max_results=4,

                start_number=
                    len(
                        sources
                    )
                    + 1,
            )


            evidence_parts.append(
                local_evidence
            )


            sources.extend(
                local_sources
            )


    # ========================================================
    # combine drawing + manuals + web evidence
    # ========================================================

    evidence = "\n\n".join(

        part

        for part
        in evidence_parts

        if part.strip()
    )


    # literally nothing was found
    if not evidence:

        return {

            "question":
                question,

            "answer":
                (
                    "I could not find technical evidence "
                    "for that yet. Upload the relevant "
                    "manual, drawing, diagram, or equipment "
                    "photo and try again."
                ),

            "sources":
                [],
        }


    # ========================================================
    # first answer pass
    # summarize whatever useful evidence we found
    # ========================================================

    refined = refine_answer(

        question,

        evidence,
    )


    # ========================================================
    # if our manuals only partially answer it,
    # quietly search manufacturer sources too
    #
    # IMPORTANT:
    # partial evidence is NOT thrown away anymore.
    # ========================================================

    if (

        local_strong

        and refined[
            "status"
        ] == "completed"

        and refined.get(
            "evidence_status"
        )
        in {
            "partial",
            "unrelated",
        }
    ):

        try:

            web_results = (
                external_search(
                    question
                )
            )


        except Exception as exc:

            print(

                "External search error "
                "after local pass:",

                type(exc).__name__,

                exc,
            )


            web_results = []


        if web_results:

            (
                web_evidence,
                web_sources,
            ) = build_web_evidence(

                web_results,

                question,

                max_results=3,

                start_number=
                    len(
                        sources
                    )
                    + 1,
            )


            combined_evidence = (

                evidence

                + "\n\n"

                + web_evidence
            )


            combined_sources = (

                sources

                + web_sources
            )


            retry = refine_answer(

                question,

                combined_evidence,
            )


            # if the second pass made a real answer,
            # use the combined evidence
            if (

                retry[
                    "status"
                ] == "completed"

                and retry.get(
                    "answer"
                )
            ):

                refined = retry

                evidence = (
                    combined_evidence
                )

                sources = (
                    combined_sources
                )


    # ========================================================
    # final answer
    #
    # DIRECT = give answer
    # PARTIAL = ALSO give answer
    # UNRELATED = only one that stops
    # ========================================================

    if (

        refined[
            "status"
        ] == "completed"

        and refined.get(
            "answer"
        )
    ):

        if (

            refined.get(
                "evidence_status"
            )
            == "unrelated"
        ):

            answer = (

                "The sources I found do not "
                "actually address this problem. "
                "Upload the exact equipment manual, "
                "drawing, diagram, or a clear photo "
                "and try again."
            )


        else:

            # this is the big change.
            # partial evidence still becomes a useful answer.
            answer = (
                refined[
                    "answer"
                ]
            )


    elif (
        refined[
            "status"
        ]
        == "not_configured"
    ):

        answer = (

            "Relevant technical evidence was found, "
            "but the answer refinement step is not "
            "configured. Add OPENAI_API_KEY to .env "
            "to generate the technician-facing summary."
        )


    else:

        answer = (

            "Relevant technical evidence was found, "
            "but EcoMaintenance could not finish the "
            "summary right now. The supporting evidence "
            "is still available below."
        )

    # ========================================================
    # exact technical fact verification
    #
    # only runs when the question involves specifications,
    # dimensions, numerical values, pins, parameters, etc.
    # ========================================================

    if (

        answer

        and evidence

        and needs_exact_fact_verification(
            question
        )

        and refined.get(
            "evidence_status"
        )
        != "unrelated"
    ):

        verification = (
            verify_exact_facts(

                question,

                evidence,

                answer,
            )
        )


        if verification.get(
            "answer"
        ):

            answer = (
                verification[
                    "answer"
                ]
            )
    # frontend only needs the clean answer and its evidence.
    # all the ugly retrieval logic stays hidden back here.

    return {

        "question":
            question,

        "answer":
            answer,

        "sources":
            sources,
    }