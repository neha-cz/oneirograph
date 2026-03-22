# Browser SPA: use `python jac.py start main.jac` so `app` registers (Jac ES export vs globalThis).
include graph.schema;
include walkers.dream_ingester;
include walkers.qubo_walker;
include walkers.adaptive_walker;
include walkers.narration_walker;

import from datetime { date }

import from parsers.dream_parser { parse_dream }

import from typing { Any }

import from api_helpers { filter_anomaly_skeletons }

import from backends.tebd_backend { report_tebd_state }

import from http { HTTPMethod }

def:priv dream_pipeline(text: str, dream_date: str) -> dict {
    parsed = parse_dream(text);
    root spawn DreamIngester(dream_data=parsed, dream_date=dream_date);
    root spawn QuboWalker(dream_date=dream_date);
    root spawn AdaptiveWalker();
    root spawn NarrationWalker(dream_date=dream_date);
    return {"status": "ok"};
}

@restspec(method=HTTPMethod.POST, path="/dream")
def:pub ingest_dream(text: str, date: str) -> dict {
    return dream_pipeline(text, date);
}

@restspec(method=HTTPMethod.GET, path="/today")
def:pub server_today() -> str {
    return str(date.today());
}

@restspec(method=HTTPMethod.GET, path="/dreams")
def:pub get_dreams() -> Any {
    anchor = root;
    return [(anchor) -->][?:DreamSession];
}

@restspec(method=HTTPMethod.GET, path="/metagraph")
def:pub get_metagraph() -> dict {
    anchor = root;
    nodes = [(anchor) -->][?:SkeletonNode];
    edge_list = [];
    for n in nodes {
        cur = n;
        for e in [(cur) -->][?:SkeletonEdge] {
            edge_list.append(
                {
                    "from": str(n),
                    "to": str(e),
                    "edge_type": str(e)
                }
            );
        }
    }
    return {"nodes": nodes, "edges": edge_list};
}

@restspec(method=HTTPMethod.GET, path="/anomalies")
def:pub get_anomalies() -> Any {
    anchor = root;
    all_nodes = [(anchor) -->][?:SkeletonNode];
    return filter_anomaly_skeletons(all_nodes);
}

@restspec(method=HTTPMethod.POST, path="/report")
def:pub create_report() -> dict {
    anchor = root;
    root spawn AdaptiveWalker();
    st = report_tebd_state();
    skels: Any = [(anchor) -->][?:SkeletonNode];
    sessions: Any = [(anchor) -->][?:DreamSession];
    st["skeleton_node_count"] = len(skels);
    st["dream_session_count"] = len(sessions);
    st["note"] = "AdaptiveWalker re-ran over current QUBO skeletons; TEBD policy buffer updated.";
    return st;
}

# SPA root for jac serve (`base_route_app = "app"`). Must live in `cl {}` so the
# server compile populates `client_manifest.exports`; a sibling `main.cl.jac` alone
# does not register with `ModuleIntrospector._load_manifest` today.
cl {
    def shell_style() -> dict {
        return {
            "minHeight": "100vh",
            "margin": "0",
            "padding": "0",
            "fontFamily": "system-ui,Segoe UI,Roboto,sans-serif",
            "background": "linear-gradient(165deg,#0c1222 0%,#1a1033 45%,#0f172a 100%)",
            "color": "#e2e8f0"
        };
    }

    def nav_btn(label, tid, tab, set_tab) -> JsxElement {
        active = tid == tab;
        bg = "#4f46e5" if active else "rgba(30,27,75,0.6)";
        return <button
            type="button"
            onClick={lambda -> None { set_tab(tid); }}
            style={{
                "padding": "10px 18px",
                "borderRadius": "999px",
                "border": "1px solid rgba(255,255,255,0.14)",
                "background": bg,
                "color": "#eef2ff",
                "cursor": "pointer",
                "fontSize": "13px",
                "fontWeight": "600" if active else "500",
                "boxShadow": "0 0 0 1px rgba(0,0,0,0.2)" if active else "none"
            }}
        >{label}</button>;
    }

    def card(inner) -> JsxElement {
        return <div style={{
            "marginTop": "24px",
            "padding": "28px",
            "borderRadius": "16px",
            "background": "rgba(15,23,42,0.55)",
            "border": "1px solid rgba(148,163,184,0.18)",
            "boxShadow": "0 24px 48px rgba(0,0,0,0.35)",
            "maxWidth": "960px"
        }}>{inner}</div>;
    }

    def pre_block(text) -> JsxElement {
        return <pre style={{
            "whiteSpace": "pre-wrap",
            "wordBreak": "break-word",
            "fontSize": "13px",
            "lineHeight": "1.55",
            "padding": "16px",
            "borderRadius": "10px",
            "background": "rgba(2,6,23,0.65)",
            "border": "1px solid rgba(51,65,85,0.5)",
            "color": "#cbd5e1",
            "maxHeight": "420px",
            "overflow": "auto"
        }}>{text}</pre>;
    }

    def lbl(t) -> JsxElement {
        return <label style={{
            "display": "block",
            "fontSize": "12px",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": "#94a3b8",
            "marginBottom": "8px",
            "fontWeight": "600"
        }}>{t}</label>;
    }

    def panel_ingest(
        dream_text,
        set_dream_text,
        dream_date,
        set_dream_date,
        status,
        busy,
        on_submit,
        on_fill_today
    ) -> JsxElement {
        return <div>
            <h2 style={{"margin": "0 0 8px 0", "fontSize": "22px", "color": "#f8fafc"}}>
                Ingest a dream
            </h2>
            <p style={{"margin": "0 0 20px 0", "color": "#94a3b8", "fontSize": "15px", "lineHeight": "1.5"}}>
                Text is parsed (OpenAI when configured), then walkers build the graph session,
                QUBO sketch, adaptive pass, and narration.
            </p>
            {lbl(t="Dream text")}
            <textarea
                value={dream_text}
                onInput={lambda (e: Any) -> None {
                    t = getattr(e, "target", e);
                    set_dream_text(getattr(t, "value", ""));
                }}
                placeholder="Describe your dream…"
                rows={10}
                style={{
                    "width": "100%",
                    "boxSizing": "border-box",
                    "padding": "14px",
                    "borderRadius": "10px",
                    "border": "1px solid rgba(148,163,184,0.35)",
                    "background": "rgba(2,6,23,0.5)",
                    "color": "#f1f5f9",
                    "fontSize": "15px",
                    "resize": "vertical",
                    "marginBottom": "16px"
                }}
            />
            <div style={{"display": "flex", "flexWrap": "wrap", "gap": "12px", "alignItems": "flex-end"}}>
                <div style={{"flex": "1", "minWidth": "200px"}}>
                    {lbl(t="Date")}
                    <input
                        type="date"
                        value={dream_date}
                        onInput={lambda (e: Any) -> None {
                            t = getattr(e, "target", e);
                            set_dream_date(getattr(t, "value", ""));
                        }}
                        style={{
                            "width": "100%",
                            "padding": "10px 12px",
                            "borderRadius": "8px",
                            "border": "1px solid rgba(148,163,184,0.35)",
                            "background": "rgba(2,6,23,0.5)",
                            "color": "#f1f5f9"
                        }}
                    />
                </div>
                <button
                    type="button"
                    onClick={on_fill_today}
                    style={{
                        "padding": "10px 16px",
                        "borderRadius": "8px",
                        "border": "1px solid rgba(148,163,184,0.35)",
                        "background": "transparent",
                        "color": "#a5b4fc",
                        "cursor": "pointer",
                        "fontWeight": "600"
                    }}
                >Use server today</button>
                <button
                    type="button"
                    disabled={busy}
                    onClick={on_submit}
                    style={{
                        "padding": "12px 24px",
                        "borderRadius": "10px",
                        "border": "none",
                        "background": "#6366f1",
                        "color": "white",
                        "cursor": "pointer" if not busy else "default",
                        "opacity": "1" if not busy else "0.55",
                        "fontWeight": "700",
                        "fontSize": "15px"
                    }}
                >{ "Run pipeline" if not busy else "Running…" }</button>
            </div>
            {(
                <p style={{
                    "marginTop": "18px",
                    "fontSize": "14px",
                    "color": "#a5b4fc" if status != "" else "transparent"
                }}>{status}</p>
            ) if status != "" else <span/>}
        </div>;
    }

    def panel_data(
        title,
        hint,
        body,
        on_refresh,
        loading,
        btn_idle,
        btn_busy
    ) -> JsxElement {
        return <div>
            <div style={{"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "16px", "flexWrap": "wrap"}}>
                <div>
                    <h2 style={{"margin": "0 0 6px 0", "fontSize": "22px", "color": "#f8fafc"}}>{title}</h2>
                    <p style={{"margin": "0", "color": "#94a3b8", "fontSize": "14px", "maxWidth": "640px"}}>{hint}</p>
                </div>
                <button
                    type="button"
                    onClick={on_refresh}
                    disabled={loading}
                    style={{
                        "padding": "10px 18px",
                        "borderRadius": "10px",
                        "border": "1px solid rgba(129,140,248,0.45)",
                        "background": "rgba(79,70,229,0.25)",
                        "color": "#e0e7ff",
                        "cursor": "pointer" if not loading else "default",
                        "opacity": "1" if not loading else "0.5",
                        "fontWeight": "600",
                        "whiteSpace": "nowrap"
                    }}
                >{ btn_busy if loading else btn_idle }</button>
            </div>
            <div style={{"marginTop": "20px"}}>{pre_block(text=body)}</div>
        </div>;
    }

    def:pub app() -> JsxElement {
        tab_p = useState("ingest");
        tab = tab_p[0];
        set_tab = tab_p[1];

        dream_p = useState("");
        dream_text = dream_p[0];
        set_dream_text = dream_p[1];

        date_p = useState("");
        dream_date = date_p[0];
        set_dream_date = date_p[1];

        ingest_status_p = useState("");
        ingest_status = ingest_status_p[0];
        set_ingest_status = ingest_status_p[1];

        busy_p = useState(False);
        busy = busy_p[0];
        set_busy = busy_p[1];

        dreams_p = useState("Press Refresh to load dream sessions from the graph root.");
        dreams_body = dreams_p[0];
        set_dreams_body = dreams_p[1];

        meta_p = useState("Press Refresh to load skeleton nodes and edges.");
        meta_body = meta_p[0];
        set_meta_body = meta_p[1];

        ano_p = useState("Press Refresh to load anomaly skeleton nodes.");
        ano_body = ano_p[0];
        set_ano_body = ano_p[1];

        ld_d_p = useState(False);
        ld_d = ld_d_p[0];
        set_ld_d = ld_d_p[1];

        ld_m_p = useState(False);
        ld_m = ld_m_p[0];
        set_ld_m = ld_m_p[1];

        ld_a_p = useState(False);
        ld_a = ld_a_p[0];
        set_ld_a = ld_a_p[1];

        rep_p = useState("Click Create report to re-run AdaptiveWalker and snapshot the TEBD policy.");
        report_body = rep_p[0];
        set_report_body = rep_p[1];

        ld_r_p = useState(False);
        ld_r = ld_r_p[0];
        set_ld_r = ld_r_p[1];

        async def fill_today() {
            try {
                t = await server_today();
                set_dream_date(str(t));
            } except Exception as e {
                set_ingest_status("Could not load server date: " + str(e));
            }
        }

        async def do_ingest() {
            if dream_text == "" {
                set_ingest_status("Please enter dream text.");
                return;
            }
            if dream_date == "" {
                set_ingest_status("Please pick a date (or Use server today).");
                return;
            }
            set_busy(True);
            set_ingest_status("");
            try {
                # Positional only: Jac ES emit resets kwargs when lowering to __jacCallFunction (would send {}).
                r = await ingest_dream(str(dream_text), str(dream_date));
                # Do not use ``r != None`` here — client lowers it to ``dict.eq(r, null)``, which calls ``Object.keys`` on null.
                set_ingest_status(
                    "Done — graph, QUBO skeleton, adaptive & narration walkers ran. (" + repr(r) + ")"
                );
                await load_dreams();
                await load_meta();
                await load_ano();
            } except Exception as e {
                set_ingest_status("Error: " + str(e));
            }
            set_busy(False);
        }

        async def load_dreams() {
            set_ld_d(True);
            try {
                r = await get_dreams();
                set_dreams_body(repr(r));
            } except Exception as e {
                set_dreams_body("Error: " + str(e));
            }
            set_ld_d(False);
        }

        async def load_meta() {
            set_ld_m(True);
            try {
                r = await get_metagraph();
                set_meta_body(repr(r));
            } except Exception as e {
                set_meta_body("Error: " + str(e));
            }
            set_ld_m(False);
        }

        async def load_ano() {
            set_ld_a(True);
            try {
                r = await get_anomalies();
                set_ano_body(repr(r));
            } except Exception as e {
                set_ano_body("Error: " + str(e));
            }
            set_ld_a(False);
        }

        async def load_report() {
            set_ld_r(True);
            try {
                r = await create_report();
                set_report_body(repr(r));
            } except Exception as e {
                set_report_body("Error: " + str(e));
            }
            set_ld_r(False);
        }

        async def boot() {
            await fill_today();
            await load_dreams();
            await load_meta();
            await load_ano();
        }

        useEffect(lambda -> None { boot(); }, []);

        def pick_main() -> JsxElement {
            if tab == "ingest" {
                return panel_ingest(
                    dream_text=dream_text,
                    set_dream_text=set_dream_text,
                    dream_date=dream_date,
                    set_dream_date=set_dream_date,
                    status=ingest_status,
                    busy=busy,
                    on_submit=lambda -> None { do_ingest(); },
                    on_fill_today=lambda -> None { fill_today(); }
                );
            } elif tab == "dreams" {
                return panel_data(
                    title="Dream sessions",
                    hint="Each ingest creates a DreamSession under root; raw text and summary live on the node.",
                    body=dreams_body,
                    on_refresh=lambda -> None { load_dreams(); },
                    loading=ld_d,
                    btn_idle="Refresh list",
                    btn_busy="Loading…"
                );
            } elif tab == "meta" {
                return panel_data(
                    title="Metagraph",
                    hint="SkeletonNode / SkeletonEdge structure produced after QUBO-side sketch (see QuboWalker).",
                    body=meta_body,
                    on_refresh=lambda -> None { load_meta(); },
                    loading=ld_m,
                    btn_idle="Refresh graph",
                    btn_busy="Loading…"
                );
            } elif tab == "anomalies" {
                return panel_data(
                    title="Anomalies",
                    hint="Skeleton nodes marked anomalous by the helper filter.",
                    body=ano_body,
                    on_refresh=lambda -> None { load_ano(); },
                    loading=ld_a,
                    btn_idle="Refresh",
                    btn_busy="Loading…"
                );
            }
            return panel_data(
                title="Create report",
                hint="POST /report re-spawns AdaptiveWalker on all skeletons and returns TEBD policy stats. Add several dreams first (Ingest dream). QML-style symbol scoring can extend backends/qnn_backend.",
                body=report_body,
                on_refresh=lambda -> None { load_report(); },
                loading=ld_r,
                btn_idle="Run report",
                btn_busy="Running…"
            );
        }

        return <div style={shell_style()}>
            <header style={{
                "padding": "28px 32px 8px 32px",
                "borderBottom": "1px solid rgba(148,163,184,0.12)"
            }}>
                <div style={{"maxWidth": "1040px", "margin": "0 auto"}}>
                    <h1 style={{
                        "margin": "0",
                        "fontSize": "clamp(28px,4vw,40px)",
                        "fontWeight": "800",
                        "letterSpacing": "-0.03em",
                        "background": "linear-gradient(90deg,#e0e7ff,#c4b5fd,#a5b4fc)",
                        "WebkitBackgroundClip": "text",
                        "color": "transparent"
                    }}>Oneirograph</h1>
                    <p style={{"margin": "10px 0 0 0", "color": "#94a3b8", "fontSize": "16px"}}>
                        Parse dream → graph symbols → QUBO skeleton → adaptive TEBD-style RL → narration; use Report after several dreams.
                    </p>
                    <nav style={{"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginTop": "22px"}}>
                        {nav_btn(label="Ingest dream", tid="ingest", tab=tab, set_tab=set_tab)}
                        {nav_btn(label="Dream sessions", tid="dreams", tab=tab, set_tab=set_tab)}
                        {nav_btn(label="Metagraph", tid="meta", tab=tab, set_tab=set_tab)}
                        {nav_btn(label="Anomalies", tid="anomalies", tab=tab, set_tab=set_tab)}
                        {nav_btn(label="Create report", tid="report", tab=tab, set_tab=set_tab)}
                    </nav>
                </div>
            </header>
            <main style={{"padding": "12px 32px 48px 32px"}}>
                <div style={{"maxWidth": "1040px", "margin": "0 auto"}}>
                    {card(inner=pick_main())}
                    <p style={{
                        "marginTop": "28px",
                        "fontSize": "12px",
                        "color": "#64748b",
                        "textAlign": "center"
                    }}>
                        API: POST /dream, /report · GET /dreams, /metagraph, /anomalies, /today · run: python jac.py start main.jac
                    </p>
                </div>
            </main>
        </div>;
    }
}
