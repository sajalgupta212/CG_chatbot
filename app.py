# app.py
import os
import re
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from agents.mapping_extractor import MappingExtractorAgent
from dotenv import load_dotenv
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

st.set_page_config(page_title="SQL Object Lineage & CRUD", layout="wide")
st.title("💬 SQL Object Lineage & CRUD Assistant (Structured + Dynamic Graphs)")

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
DEFAULT_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
DEFAULT_USER = os.getenv("SNOWFLAKE_USER", "")
DEFAULT_ROLE = os.getenv("SNOWFLAKE_ROLE", "")
DEFAULT_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "")
DEFAULT_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "")
DEFAULT_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "")
DEFAULT_PRIVATE_KEY_FILE = os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE", "")
DEFAULT_PRIVATE_KEY_PASSPHRASE = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
# We will re-read this at call time
GROQ_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# Sidebar - Snowflake Config
# -----------------------------
st.sidebar.header("Snowflake Config")
sf_account = st.sidebar.text_input("Account", value=DEFAULT_ACCOUNT)
sf_user = st.sidebar.text_input("User", value=DEFAULT_USER)
sf_role = st.sidebar.text_input("Role", value=DEFAULT_ROLE)
sf_warehouse = st.sidebar.text_input("Warehouse", value=DEFAULT_WAREHOUSE)
sf_database = st.sidebar.text_input("Database", value=DEFAULT_DATABASE)
sf_schema = st.sidebar.text_input("Schema", value=DEFAULT_SCHEMA)
sf_private_key_file = st.sidebar.text_input("Private Key File", value=DEFAULT_PRIVATE_KEY_FILE)
sf_private_key_passphrase = st.sidebar.text_input("Private Key Passphrase", type="password", value=DEFAULT_PRIVATE_KEY_PASSPHRASE)

st.sidebar.markdown("---")
st.sidebar.header("Options")
show_lineage = st.sidebar.checkbox("Show Lineage Graph", True)
show_usage_matrix = st.sidebar.checkbox("Show CRUD Usage Matrix", True)
physics_strength = st.sidebar.slider("Graph Physics Strength", 0.5, 5.0, 1.0)

# -----------------------------
# Session state
# -----------------------------
if "agent" not in st.session_state: st.session_state.agent = None
if "global_graph" not in st.session_state: st.session_state.global_graph = None
if "node_sql_map" not in st.session_state: st.session_state.node_sql_map = {}
if "crud_matrix" not in st.session_state: st.session_state.crud_matrix = {}
if "objects" not in st.session_state: st.session_state.objects = {"PROCEDURE": [], "VIEW": [], "TABLE": []}

# -----------------------------
# Connect to Snowflake
# -----------------------------
if st.sidebar.button("Connect"):
    try:
        st.session_state.agent = MappingExtractorAgent()
        if st.session_state.agent.conn:
            st.success("✅ Connected to Snowflake!")
        else:
            st.error("❌ Connection failed")
    except Exception as e:
        st.error(f"❌ Error: {e}")

agent = st.session_state.agent

# -----------------------------
# Display basic Snowflake info
# -----------------------------
if agent and agent.conn:
    try:
        cur = agent.conn.cursor()
        cur.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE(), CURRENT_ROLE()")
        db, schema, wh, role = cur.fetchone()
        cur.close()
        st.sidebar.markdown("### ❗ Snowflake Info")
        st.sidebar.write(f"**Database:** {db}")
        st.sidebar.write(f"**Schema:** {schema}")
        st.sidebar.write(f"**Warehouse:** {wh}")
        st.sidebar.write(f"**Role:** {role}")
    except Exception as e:
        st.sidebar.error(f"Error fetching Snowflake info: {e}")

# -----------------------------
# Fetch objects from DB
# -----------------------------
def fetch_objects(agent, database, schema, obj_type):
    if not agent or not agent.conn:
        return []
    query = f"""
        SELECT OBJECT_NAME, DDL_TEXT
        FROM "{database}"."{schema}"."DDL_METADATA"
        WHERE OBJECT_DOMAIN = '{obj_type}'
    """
    cur = agent.conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    return [{"name": r[0].upper(), "ddl": r[1]} for r in rows]

def extract_objects_from_ddl(ddl_text):
    if not ddl_text:
        return []
    ddl_text = re.sub(r"--.*", "", ddl_text)
    ddl_text = re.sub(r"\s+", " ", ddl_text)
    return [t.upper() for t in re.findall(r"(?:FROM|JOIN|INSERT INTO|UPDATE|DELETE FROM)\s+([^\s(]+)", ddl_text, re.IGNORECASE)]

# -----------------------------
# Build object-level graph
# -----------------------------
def build_graph(objects_list):
    G = nx.DiGraph()
    node_sql_map = {}
    for obj in objects_list:
        obj_name = obj["name"]
        ddl_text = obj["ddl"] or ""
        refs = extract_objects_from_ddl(ddl_text)
        if not G.has_node(obj_name):
            G.add_node(obj_name)
        node_sql_map.setdefault(obj_name, []).append(ddl_text)
        for ref in refs:
            if not G.has_node(ref):
                G.add_node(ref)
            G.add_edge(obj_name, ref)
            node_sql_map.setdefault(ref, []).append(ddl_text)
    return G, node_sql_map

# -----------------------------
# Helper: node type mapping
# -----------------------------
def build_node_type_map(objects_dict):
    """
    Build mapping of node name -> type ("TABLE"/"VIEW"/"PROCEDURE"/"UNKNOWN")
    """
    m = {}
    for typ in ["TABLE", "VIEW", "PROCEDURE"]:
        for o in objects_dict.get(typ, []):
            m[o["name"]] = typ
    return m

def node_color_by_type(node_name, node_type_map):
    t = node_type_map.get(node_name, "UNKNOWN")
    # color hex codes chosen for good contrast
    if t == "TABLE":
        return "#1976D2"  # blue
    if t == "VIEW":
        return "#2E7D32"  # green
    if t == "PROCEDURE":
        return "#F57C00"  # orange
    return "#9E9E9E"      # gray

# -----------------------------
# Fetch CRUD usage with timestamps
# -----------------------------
def fetch_crud_usage(agent, database, schema, lookback_days=30):
    if not agent or not agent.conn:
        return {}
    start_time = datetime.utcnow() - timedelta(days=lookback_days)
    query = f"""
        SELECT QUERY_TEXT, START_TIME
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE DATABASE_NAME = '{database}'
          AND SCHEMA_NAME = '{schema}'
          AND START_TIME >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'
        ORDER BY START_TIME DESC
    """
    cur = agent.conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()

    crud_matrix = {}
    for query_text, ts in rows:
        q = query_text.upper()
        for t in re.findall(r"INSERT INTO\s+([^\s(]+)", q): crud_matrix.setdefault(t, {}).setdefault("C", []).append(ts)
        for t in re.findall(r"UPDATE\s+([^\s,]+)", q): crud_matrix.setdefault(t, {}).setdefault("U", []).append(ts)
        for t in re.findall(r"DELETE FROM\s+([^\s,]+)", q): crud_matrix.setdefault(t, {}).setdefault("D", []).append(ts)
        for t in re.findall(r"FROM\s+([^\s,;]+)", q) + re.findall(r"JOIN\s+([^\s,;]+)", q): crud_matrix.setdefault(t, {}).setdefault("R", []).append(ts)

    result = {}
    for obj, ops in crud_matrix.items():
        result[obj] = {
            "Object": obj,
            "C": len(ops.get("C", [])),
            "U": len(ops.get("U", [])),
            "D": len(ops.get("D", [])),
            "R": len(ops.get("R", [])),
            "Execution Timestamps": {k: ops[k] for k in ops},
            "Total CRUD": sum(len(v) for v in ops.values())
        }
    return result

# -----------------------------
# Load DB objects, graph, and CRUD
# -----------------------------
if agent and agent.conn:
    if st.sidebar.button("Load Graph & CRUD"):
        procs = fetch_objects(agent, sf_database, sf_schema, "PROCEDURE")
        views = fetch_objects(agent, sf_database, sf_schema, "VIEW")
        tables = fetch_objects(agent, sf_database, sf_schema, "TABLE")
        all_objects = procs + views + tables
        G, node_sql_map = build_graph(all_objects)
        st.session_state.global_graph = G
        st.session_state.node_sql_map = node_sql_map
        st.session_state.objects = {"PROCEDURE": procs, "VIEW": views, "TABLE": tables}
        st.session_state.crud_matrix = fetch_crud_usage(agent, sf_database, sf_schema)
        st.success(f"✅ Graph loaded with {len(G.nodes)} objects. CRUD fetched for {len(st.session_state.crud_matrix)} objects.")

# -----------------------------
# Sidebar - Display DB Objects
# -----------------------------
if st.session_state.objects:
    st.sidebar.subheader("Database Objects")
    for obj_type in ["TABLE", "VIEW", "PROCEDURE"]:
        objs = [o["name"] for o in st.session_state.objects.get(obj_type, [])]
        st.sidebar.write(f"**{obj_type}s ({len(objs)}):** {', '.join(objs[:10])}{' ...' if len(objs)>10 else ''}")

# -----------------------------
# Lineage Graph (Main)
# -----------------------------
if show_lineage and st.session_state.global_graph:
    st.subheader("🌐 Object-Level Lineage Graph")
    obj_type = st.selectbox("Select Object Type", ["PROCEDURE", "VIEW", "TABLE"])
    obj_list = [o["name"] for o in st.session_state.objects.get(obj_type, [])]
    selected_obj = st.selectbox(f"Select {obj_type}", [""] + obj_list)

    if selected_obj:
        G = st.session_state.global_graph
        node_sql_map = st.session_state.node_sql_map
        net = Network(height="650px", width="100%", directed=True)
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])

        # build node type map for coloring
        node_type_map = build_node_type_map(st.session_state.objects)

        for node in net.nodes:
            n = node["id"]
            node["color"] = node_color_by_type(n, node_type_map)
            # highlight selected
            if n == selected_obj:
                node["color"] = "#FFB300"  # brighter orange for selected
                node["size"] = 28

        for edge in net.edges:
            if edge["from"] == selected_obj or edge["to"] == selected_obj:
                edge["color"] = "red"
                edge["width"] = 3

        for node in net.nodes:
            sql_snippets = node_sql_map.get(node["id"], [])
            node["title"] = "\n\n".join(sql_snippets[:3])

        tmp_path = "temp_graph.html"
        net.save_graph(tmp_path)
        HtmlFile = open(tmp_path, 'r', encoding='utf-8').read()
        components.html(HtmlFile, height=650, scrolling=True)

# -----------------------------
# Dynamic CRUD Matrix
# -----------------------------
if st.session_state.crud_matrix and show_usage_matrix:
    st.subheader("📊 Dynamic CRUD Usage Matrix")

    # Sidebar filters
    obj_type_filter = st.sidebar.multiselect(
        "Object Type", ["TABLE", "VIEW", "PROCEDURE"], default=["TABLE", "VIEW", "PROCEDURE"]
    )
    crud_type_filter = st.sidebar.multiselect(
        "CRUD Type", ["C", "U", "D", "R"], default=["C", "U", "D", "R"]
    )
    start_date = st.sidebar.date_input("Start Date", value=datetime.utcnow() - timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", value=datetime.utcnow())

    # Filtered data
    filtered_objects = []
    for obj_name, ops in st.session_state.crud_matrix.items():
        obj_type = "TABLE"  # default
        for t, lst in st.session_state.objects.items():
            if any(o["name"] == obj_name for o in lst):
                obj_type = t
                break
        if obj_type not in obj_type_filter:
            continue

        filtered_ops = {}
        for crud, times in ops.get("Execution Timestamps", {}).items():
            if crud not in crud_type_filter:
                continue
            filtered_ops[crud] = [ts for ts in times if start_date <= ts.date() <= end_date]

        if filtered_ops:
            filtered_objects.append({
                "Object": obj_name,
                "Type": obj_type,
                **{k: len(v) for k, v in filtered_ops.items()},
                "Total CRUD": sum(len(v) for v in filtered_ops.values()),
                "Execution Timestamps": filtered_ops
            })

    if filtered_objects:
        df = pd.DataFrame(filtered_objects)
        st.dataframe(df)
    else:
        st.info("No CRUD activity found for selected filters.")

# -----------------------------
# Structured Chatbot Q&A + Dynamic Graph
# -----------------------------
st.subheader("💬 Ask Questions about SQL Lineage / CRUD (Structured Answer)")
question = st.text_area("Enter your question:")

SYSTEM_SCHEMA_PROMPT = """
You are an expert SQL lineage & CRUD analyst. You MUST reply strictly in the following MARKDOWN structure with the section headers exactly as shown. Do NOT add any other keys or commentary outside this structure.

### 🔍 Understanding
Short interpretation of what the user asked.

### 📘 Relevant Context Used
- List object names (TABLE/VIEW/PROCEDURE)
- CRUD mentions and short DDL references or snippets used

### 🧠 Reasoning
Step-by-step reasoning that leads to the final answer (concise).

### ✅ Final Answer
One-paragraph final answer.

### 📌 References
- Objects involved (list)
- CRUD operations (map)
- Lineage steps (list)
"""

def call_groq_structured(prompt):
    """
    Call Groq LLM enforcing the SYSTEM_SCHEMA_PROMPT, return raw text.
    """
    groq_key = os.getenv("GROQ_API_KEY", None)
    if not groq_key:
        st.error("GROQ_API_KEY missing in environment. Set it in .env or reload env.")
        return None
    try:
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_SCHEMA_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Groq call failed: {e}")
        return None

def extract_entities_from_text(text):
    """
    Try to extract object-like tokens (UPPERCASE, with underscores/dots) from LLM text.
    Returns unique list preserving order.
    """
    if not text:
        return []
    tokens = re.findall(r"[A-Z0-9_.]{3,}", text)
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def build_subgraph_from_entities(entities, full_graph, depth=1):
    """
    Build subgraph containing entities and their immediate neighbors up to 'depth' hops.
    Returns subgraph (nx.DiGraph) and list of nodes highlighted.
    """
    if not full_graph or not entities:
        return None, []
    sub_nodes = set()
    for e in entities:
        if e in full_graph:
            sub_nodes.add(e)
            # BFS up to depth
            frontier = {e}
            for _ in range(depth):
                next_frontier = set()
                for n in frontier:
                    next_frontier.update(set(full_graph.successors(n)))
                    next_frontier.update(set(full_graph.predecessors(n)))
                frontier = next_frontier - sub_nodes
                sub_nodes.update(frontier)
    subG = full_graph.subgraph(sub_nodes).copy() if sub_nodes else None
    return subG, list(sub_nodes)

def render_pyvis_graph(G, highlight_nodes=None, node_type_map=None, height="500px"):
    """
    Render a pyvis graph with node color-coding and highlighting.
    """
    if G is None or len(G.nodes) == 0:
        return None

    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.from_nx(G)
    # apply color coding via node_type_map
    for node in net.nodes:
        n = node["id"]
        color = node_color_by_type(n, node_type_map or {})
        node["color"] = color
        node["title"] = node.get("title", "")
        node["label"] = n

    # highlight nodes
    if highlight_nodes:
        for node in net.nodes:
            if node["id"] in highlight_nodes:
                node["color"] = "#FFB300"  # highlight color
                node["size"] = 26

    # highlight edges adjacent to highlighted nodes
    for edge in net.edges:
        if highlight_nodes and (edge["from"] in highlight_nodes or edge["to"] in highlight_nodes):
            edge["color"] = "red"
            edge["width"] = 3

    tmp = "temp_subgraph.html"
    net.save_graph(tmp)
    html = open(tmp, 'r', encoding='utf-8').read()
    return html

if st.button("Ask"):
    if not agent or not agent.conn:
        st.warning("Connect to Snowflake first")
    else:
        # Build context lines: DDL snippets + CRUD rows (stringified)
        context_lines = []
        for obj_type in ["PROCEDURE", "VIEW", "TABLE"]:
            for obj in st.session_state.objects.get(obj_type, []):
                if obj.get("ddl"):
                    context_lines.append(f"{obj_type} {obj['name']} DDL:\n{obj['ddl']}\n")
        for obj_name, ops in st.session_state.crud_matrix.items():
            context_lines.append(f"Object {obj_name} CRUD:\n{json.dumps(ops, default=str)}\n")

        if not context_lines:
            st.warning("No context available (no DDLs or CRUD). Load Graph & CRUD first.")
        else:
            # try to pick the single most relevant context block using embeddings
            try:
                embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                qvec = embedder.encode([question], convert_to_numpy=True)[0]
                vecs = embedder.encode(context_lines, convert_to_numpy=True)
                if vecs is None or vecs.size == 0 or vecs.shape[0] == 0:
                    top_context = "\n".join(context_lines[:3])
                else:
                    # safe shapes
                    qvec = np.asarray(qvec)
                    vecs = np.asarray(vecs)
                    if vecs.ndim == 1:
                        sims = np.dot(vecs, qvec)
                        best_idx = 0
                    else:
                        sims = np.dot(vecs, qvec)
                        best_idx = int(np.argmax(sims))
                    top_context = context_lines[best_idx]
            except Exception as e:
                st.warning(f"Embedding step failed; falling back to first contexts: {e}")
                top_context = "\n".join(context_lines[:3])

            prompt = f"CONTEXT:\n{top_context}\n\nQUESTION: {question}"
            raw_answer = call_groq_structured(prompt)

            if not raw_answer:
                st.error("No answer returned from LLM.")
            else:
                # Display structured answer exactly as returned by LLM
                st.markdown(raw_answer)

                # Extract entities mentioned in the answer (object names, uppercase tokens)
                entities = extract_entities_from_text(raw_answer)

                # Build node type map from current objects
                node_type_map = build_node_type_map(st.session_state.objects)

                # Build subgraph around entities
                subG, highlight_nodes = build_subgraph_from_entities(entities, st.session_state.global_graph, depth=1)

                if subG is not None and len(subG.nodes) > 0:
                    st.subheader("🕸 Dynamic Graph based on Chatbot Answer")
                    html = render_pyvis_graph(subG, highlight_nodes=highlight_nodes, node_type_map=node_type_map, height="520px")
                    if html:
                        components.html(html, height=520, scrolling=True)
                    else:
                        st.info("No dynamic graph to show.")
                else:
                    st.info("No related objects found in the main graph for the entities mentioned in the answer.")

# -----------------------------
# Notes
# -----------------------------
st.markdown("---")
st.markdown("**Notes & tips**\n- Change GROQ_API_KEY in your `.env` and reload environment if needed.\n- Color coding: TABLE (blue), VIEW (green), PROCEDURE (orange).")

