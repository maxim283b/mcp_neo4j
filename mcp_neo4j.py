import asyncio
import os
import logging
from typing import List, Dict, Any, Literal, Optional, Tuple

from fastmcp import FastMCP
from rapidfuzz import fuzz
from neo4j import GraphDatabase, Driver
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


# =========================
# LOGGING
# =========================
logger = logging.getLogger("neo4j_mcp_server")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# =========================
# MCP (MUST be top-level)
# =========================
mcp = FastMCP("neo4j-mcp")
logger.info("MCP server object created")

# =========================
# NEO4J (LAZY INIT — NO NETWORK ON IMPORT)
# =========================
_driver: Optional[Driver] = None


def get_driver() -> Driver:
    """
    Create Neo4j driver only when a tool is actually called.
    This avoids FastMCP Cloud build/inspect/pre-flight DNS/network failures.
    """
    global _driver
    if _driver is not None:
        return _driver

    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_PASSWORD")
    user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "neo4j"

    if not uri:
        raise RuntimeError("Missing env var: NEO4J_URI")
    if not password:
        raise RuntimeError("Missing env var: NEO4J_PASSWORD")

    # IMPORTANT: do NOT verify connectivity / do NOT run queries here.
    _driver = GraphDatabase.driver(uri, auth=(user, password))
    logger.info("Neo4j driver initialized (lazy)")
    return _driver


def close_driver():
    global _driver
    if _driver is not None:
        try:
            _driver.close()
        except Exception:
            pass
        _driver = None


# =========================
# HELPERS
# =========================
def normalize_keywords(val: Any) -> List[str]:
    """
    Accept:
      - ["a", "b"]
      - "a,b"
      - "a"
    Return list of clean strings.
    """
    if val is None:
        return []
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    if isinstance(val, list):
        out: List[str] = []
        for v in val:
            s = str(v).strip()
            if not s:
                continue
            out.extend([x.strip() for x in s.split(",") if x.strip()])
        return out
    s = str(val).strip()
    return [s] if s else []


def normalize_limit(limit: Any, default: int = 10) -> int:
    try:
        v = int(limit)
    except Exception:
        v = default
    return max(1, min(v, 50))


def _run(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    driver = get_driver()
    with driver.session() as s:
        return [r.data() for r in s.run(query, params)]


def _tags_to_list(v: Any) -> List[str]:
    """
    Tags in DB can be:
      - list
      - string with '.' separated tokens
      - string with ',' separated tokens
    We normalize into list[str].
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    parts: List[str] = []
    for chunk in s.split("."):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([x.strip() for x in chunk.split(",") if x.strip()])
    return parts


def _split_flags(v: Any) -> List[str]:
    """
    Your DB stores requires/Produces as STRING (often comma-separated),
    but support list too.
    """
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for x in v:
            out.extend([t.strip() for t in str(x).split(",") if t.strip()])
        return out
    return [t.strip() for t in str(v).split(",") if t.strip()]


# =========================
# SEARCH COMMANDS (tool #1)
# =========================
def fetch_commands_for_search() -> List[Dict[str, Any]]:
    """
    Collect searchable text for each command.
    Graph: (steps)-[:has_command]->(commands)-[:has_params]->(Params)
    """
    query = """
    MATCH (c:commands)
    OPTIONAL MATCH (s:steps)-[:has_command]->(c)
    OPTIONAL MATCH (c)-[:has_params]->(p:Params)
    RETURN
      elementId(c) AS command_id,
      properties(c) AS command_props,
      COALESCE(c.description, '') AS command_desc,
      COALESCE(s.description, '') AS step_desc,
      COALESCE(p.Description, '') AS param_desc,
      COALESCE(c.Tags, []) AS command_tags,
      COALESCE(s.Tags, []) AS step_tags,
      COALESCE(p.Tags, []) AS param_tags
    """

    rows: List[Dict[str, Any]] = []
    for r in _run(query, {}):
        text = " ".join(
            [
                str(r.get("command_desc") or ""),
                str(r.get("step_desc") or ""),
                str(r.get("param_desc") or ""),
                *_tags_to_list(r.get("command_tags")),
                *_tags_to_list(r.get("step_tags")),
                *_tags_to_list(r.get("param_tags")),
            ]
        )
        rows.append(
            {
                "command_id": r["command_id"],
                "command_props": r["command_props"],
                "text": text,
            }
        )
    return rows


def score_commands(rows: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    scored: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        cid = row["command_id"]
        props = row["command_props"]
        text = (row.get("text") or "").lower()

        score = 0.0
        for kw in keywords:
            kw_l = kw.lower().strip()
            if not kw_l:
                continue
            score += float(fuzz.partial_ratio(kw_l, text))
            if kw_l in text.split():
                score += 50.0

        if cid not in scored:
            scored[cid] = {"command_id": cid, "command_props": props, "score": 0.0}
        scored[cid]["score"] += score

    out = list(scored.values())
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


@mcp.tool(description="Search commands by keywords (fuzzy)")
async def search_command(
    keywords: Optional[List[str]] = None,
    limit: int = 10,

    # n8n / client fields ignored (explicit; **kwargs NOT allowed)
    request: Optional[str] = None,
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None,
    tool: Optional[str] = None,
):
    if not keywords or (isinstance(keywords, list) and len(keywords) == 0):
        keywords = normalize_keywords(request) if request else []
    keywords = normalize_keywords(keywords)
    limit = normalize_limit(limit, default=10)

    if not keywords:
        return {"keywords": [], "matched_commands": []}

    rows = await asyncio.to_thread(fetch_commands_for_search)
    scored = await asyncio.to_thread(score_commands, rows, keywords)

    return {"keywords": keywords, "matched_commands": scored[:limit]}


# =========================
# BUILD CHAIN (tool #2)
# =========================
def get_command_props(command_id: str) -> Dict[str, Any]:
    q = """
    MATCH (c:commands)
    WHERE elementId(c) = $cid
    RETURN properties(c) AS p
    """
    rows = _run(q, {"cid": command_id})
    return (rows[0].get("p") if rows else {}) or {}


def get_steps_for_command(command_id: str) -> List[Dict[str, Any]]:
    q = """
    MATCH (s:steps)-[:has_command]->(c:commands)
    WHERE elementId(c) = $cid
    RETURN elementId(s) AS step_id, properties(s) AS step_props
    """
    return _run(q, {"cid": command_id})


def choose_best_step_for_command(command_id: str, command_props: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    If command linked to multiple steps:
      - prefer step where step_props.Step == command_props.Step
      - else first
    """
    steps = get_steps_for_command(command_id)
    if not steps:
        return None

    desired = (command_props or {}).get("Step")
    if desired:
        for st in steps:
            sp = st.get("step_props") or {}
            if sp.get("Step") == desired:
                return st
    return steps[0]


def get_commands_for_step(step_id: str) -> List[Dict[str, Any]]:
    """
    Return ALL commands for a step + their params.
    DISTINCT avoids duplicates due to optional joins.
    """
    q = """
    MATCH (s:steps)-[:has_command]->(c:commands)
    WHERE elementId(s) = $sid
    WITH DISTINCT c
    OPTIONAL MATCH (c)-[:has_params]->(p:Params)
    RETURN
      elementId(c) AS command_id,
      properties(c) AS command_props,
      collect(DISTINCT CASE
        WHEN p IS NULL THEN NULL
        ELSE {id: elementId(p), props: properties(p)}
      END) AS params
    """
    rows = _run(q, {"sid": step_id})
    for r in rows:
        r["params"] = [x for x in (r.get("params") or []) if x is not None]
    return rows


def get_requires(step_props: Dict[str, Any]) -> List[str]:
    req = step_props.get("requires")
    if req is None:
        req = step_props.get("require")
    return _split_flags(req)


def get_producers(flag: str) -> List[Dict[str, Any]]:
    """
    Produces/produses are STRING in your DB -> convert to list in Cypher.
    """
    q = """
    MATCH (s:steps)
    WITH s,
         CASE
           WHEN s.Produces IS NULL THEN []
           WHEN valueType(s.Produces) = 'LIST' THEN s.Produces
           ELSE [x IN split(toString(s.Produces), ',') | trim(x)]
         END AS prod1,
         CASE
           WHEN s.produces IS NULL THEN []
           WHEN valueType(s.produces) = 'LIST' THEN s.produces
           ELSE [x IN split(toString(s.produces), ',') | trim(x)]
         END AS prod2
    WHERE $flag IN prod1 OR $flag IN prod2
    RETURN elementId(s) AS step_id, properties(s) AS step_props
    """
    return _run(q, {"flag": flag})


def build_chain(step_id: str, step_props: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    DFS: step.requires -> producers(flag)
    Output is topologically ordered prerequisites, ending with target step.
    """
    visited = set()
    ordered: List[Tuple[str, Dict[str, Any]]] = []

    def dfs(sid: str, sprops: Dict[str, Any]):
        if sid in visited:
            return
        visited.add(sid)

        for req in get_requires(sprops):
            for prod in get_producers(req):
                dfs(prod["step_id"], prod["step_props"])

        ordered.append((sid, sprops))

    dfs(step_id, step_props)
    return ordered


async def _build_chain_for_command(command_id: str, include_params: bool) -> Dict[str, Any]:
    cmd_props = await asyncio.to_thread(get_command_props, command_id)
    step = await asyncio.to_thread(choose_best_step_for_command, command_id, cmd_props)
    if not step:
        return {"error": f"Step not found for command_id={command_id}", "chain": []}

    chain_steps = await asyncio.to_thread(build_chain, step["step_id"], step["step_props"])

    chain: List[Dict[str, Any]] = []
    for sid, sprops in chain_steps:
        cmds = await asyncio.to_thread(get_commands_for_step, sid)
        if not include_params:
            cmds = [{"command_id": c["command_id"], "command_props": c["command_props"]} for c in cmds]
        chain.append({"step_id": sid, "step_props": sprops, "commands": cmds})

    return {"target_command_id": command_id, "target_step_id": step["step_id"], "chain": chain}


@mcp.tool(description="Build prerequisite step chain for a selected command_id (includes all commands in each step)")
async def build_chain(
    command_id: str,
    include_params: bool = True,

    # ignored
    id: Optional[str] = None,
    toolCallId: Optional[str] = None,
    tool: Optional[str] = None,
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,

    session_id: Optional[str] = None,
    user_message: Optional[str] = None,
    state: Optional[str] = None,
    params: Optional[dict] = None,
    chain: Optional[Any] = None,
    matched_command: Optional[Any] = None,
    timestamp: Optional[str] = None,
):
    return await _build_chain_for_command(command_id=command_id, include_params=include_params)


# =========================
# DEBUGGER (tool #3)
# =========================
@mcp.tool(description="Debug: show types/values of requires/Produces for steps")
async def debug_steps_requires_produces(
    limit: int = 50,

    # ignored
    request: Optional[str] = None,
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None,
    tool: Optional[str] = None,
):
    limit = normalize_limit(limit, default=50)
    q = f"""
    MATCH (s:steps)
    WHERE s.requires IS NOT NULL OR s.Produces IS NOT NULL OR s.require IS NOT NULL OR s.produces IS NOT NULL
    RETURN
      s.Step AS Step,
      s.requires AS requires,
      valueType(s.requires) AS requires_type,
      s.Produces AS Produces,
      valueType(s.Produces) AS Produces_type,
      s.require AS require,
      valueType(s.require) AS require_type,
      s.produces AS produces,
      valueType(s.produces) AS produces_type
    LIMIT {limit}
    """
    return {"rows": _run(q, {})}


# =========================
# SERVER START
# =========================
async def main(
    transport: Literal["http", "stdio", "sse"] = "http",
    host: str = "0.0.0.0",
    port: int = int(os.getenv("PORT", "8080")),
    path: str = "/mcp/",
    allow_origins: Optional[List[str]] = None,
):
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=allow_origins or ["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    ]

    logger.info(f"Starting Neo4j MCP server on {host}:{port}{path} transport={transport}")

    if transport == "http":
        await mcp.run_http_async(host=host, port=port, path=path, middleware=middleware)
    elif transport == "sse":
        await mcp.run_http_async(host=host, port=port, path=path, middleware=middleware, transport="sse")
    elif transport == "stdio":
        await mcp.run_stdio_async()
    else:
        raise ValueError(f"Unsupported transport: {transport}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        close_driver()
