import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FileText, Info, BookOpen, Download, MapPin, Layers, Sprout, Mountain, Database, FileDown } from 'lucide-react';

interface LocationData {
    address?: string;
    address_display?: string;
    water_status?: string;
    water_body_status?: string;
    elevation?: number;
    country?: string;
    lat?: number;
    lon?: number;
    is_inland_water?: boolean;
}

interface InputParams {
    lat?: number;
    lon?: number;
    elevation?: number;
    current_land_use?: string;
    soil?: string;
    biodiv?: string;
    distance_to_coastline?: number;
    location_str?: string;
    location_str_for_print?: string;
    downloadable_datasets?: Array<{ path: string; label: string; source: string }>;
    [key: string]: unknown;
}

interface Props {
    report: string;
    plots: string[];
    location: LocationData | null;
    running: boolean;
    inputParams?: InputParams;
    references?: { used?: string[] };
    sessionId?: string;
}

type TabKey = 'report' | 'additional' | 'data' | 'references';

export function ReportView({ report, plots, location, running, inputParams, references, sessionId }: Props) {
    const [activeTab, setActiveTab] = useState<TabKey>('report');

    if (!report && !running) {
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '60px 20px',
                textAlign: 'center',
                color: 'var(--text-muted)',
                gap: 12,
            }}>
                <MapPin size={40} style={{ opacity: 0.3 }} />
                <h2 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Select a location &amp; ask a question
                </h2>
                <p style={{ maxWidth: 400, lineHeight: 1.6, fontSize: '0.85rem' }}>
                    Click anywhere on the map to select coordinates, type your climate
                    question, and ClimSight will analyze the location using climate
                    models, environmental data, and AI-powered insights.
                </p>
            </div>
        );
    }

    if (running && !report) {
        return (
            <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-secondary)' }}>
                <div className="spinner" style={{ margin: '0 auto 12px' }} />
                <p style={{ fontSize: '0.85rem' }}>Analyzing location…</p>
            </div>
        );
    }

    const tabs: { key: TabKey; label: string; icon: React.ReactNode }[] = [
        { key: 'report', label: 'Report', icon: <FileText size={13} /> },
        { key: 'additional', label: 'Figures', icon: <Info size={13} /> },
        { key: 'data', label: 'Data', icon: <Database size={13} /> },
        { key: 'references', label: 'References', icon: <BookOpen size={13} /> },
    ];

    const usedRefs = references?.used ?? [];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0, flex: 1 }}>
            {/* Location banner */}
            {location && (
                <div className="glass" style={{ padding: '8px 14px', marginBottom: 8, fontSize: '0.8rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--accent)' }}>
                        <MapPin size={13} />
                        <strong>{location.address_display || location.address || `${location.lat?.toFixed(4)}, ${location.lon?.toFixed(4)}`}</strong>
                    </div>
                    {location.is_inland_water && location.water_body_status && (
                        <div className="badge badge-warning" style={{ marginTop: 4 }}>
                            ⚠️ {location.water_body_status}: Analyses are designed for land areas.
                        </div>
                    )}
                </div>
            )}

            {/* Tabs */}
            <div style={{
                display: 'flex',
                gap: 0,
                borderBottom: '1px solid var(--border)',
                marginBottom: 12,
            }}>
                {tabs.map((tab) => (
                    <button
                        key={tab.key}
                        onClick={() => setActiveTab(tab.key)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 5,
                            padding: '8px 16px',
                            background: 'transparent',
                            border: 'none',
                            borderBottom: activeTab === tab.key ? '2px solid var(--text-primary)' : '2px solid transparent',
                            color: activeTab === tab.key ? 'var(--text-primary)' : 'var(--text-muted)',
                            cursor: 'pointer',
                            fontSize: '0.8rem',
                            fontWeight: activeTab === tab.key ? 500 : 400,
                            letterSpacing: '0.02em',
                            fontFamily: 'var(--font)',
                            transition: 'all 0.2s',
                        }}
                    >
                        {tab.icon}
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab content */}
            <div style={{ flex: 1, overflow: 'auto' }}>
                {activeTab === 'report' && (
                    <>
                        <div className="report-content">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
                        </div>

                        {/* Plots moved to Figures tab only */}

                        {/* Download */}
                        {report && (
                            <div style={{
                                display: 'flex',
                                gap: 8,
                                marginTop: 20,
                                paddingTop: 16,
                                borderTop: '1px solid var(--border)',
                            }}>
                                <button
                                    className="btn-primary"
                                    style={{ fontSize: '0.78rem', padding: '8px 14px' }}
                                    onClick={() => downloadText(report)}
                                >
                                    <Download size={13} style={{ marginRight: 4, verticalAlign: -2 }} />
                                    Download Text
                                </button>
                                <button
                                    className="btn-primary"
                                    style={{ fontSize: '0.78rem', padding: '8px 14px' }}
                                    onClick={() => downloadMarkdown(report)}
                                >
                                    <FileDown size={13} style={{ marginRight: 4, verticalAlign: -2 }} />
                                    Download Markdown
                                </button>
                            </div>
                        )}
                    </>
                )}

                {activeTab === 'additional' && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: '0.85rem' }}>
                        {inputParams?.lat != null && (
                            <InfoRow icon={<MapPin size={13} />} label="Coordinates" value={`${safe(inputParams.lat)}, ${safe(inputParams.lon)}`} />
                        )}
                        {inputParams?.elevation != null && (
                            <InfoRow icon={<Mountain size={13} />} label="Elevation" value={`${safe(inputParams.elevation)} m`} />
                        )}
                        {inputParams?.current_land_use && (
                            <InfoRow icon={<Layers size={13} />} label="Land use" value={safe(inputParams.current_land_use)} />
                        )}
                        {inputParams?.soil && (
                            <InfoRow icon={<Layers size={13} />} label="Soil type" value={safe(inputParams.soil)} />
                        )}
                        {inputParams?.biodiv && (
                            <InfoRow icon={<Sprout size={13} />} label="Occurring species" value={safe(inputParams.biodiv)} />
                        )}
                        {inputParams?.distance_to_coastline != null && (
                            <InfoRow icon={<MapPin size={13} />} label="Distance to shore" value={formatDist(inputParams.distance_to_coastline)} />
                        )}

                        {/* Categorized plots */}
                        {plots.length > 0 && (
                            <CategorizedPlots plots={plots} />
                        )}

                        {!inputParams && plots.length === 0 && (
                            <p style={{ color: 'var(--text-muted)' }}>
                                Additional information will appear after running an analysis.
                            </p>
                        )}
                    </div>
                )}

                {activeTab === 'data' && (
                    <DataTab datasets={inputParams?.downloadable_datasets ?? []} sessionId={sessionId} />
                )}

                {activeTab === 'references' && (
                    <div style={{ fontSize: '0.85rem' }}>
                        {usedRefs.length > 0 ? (
                            <ul style={{ paddingLeft: 18, color: 'var(--text-secondary)' }}>
                                {usedRefs.map((ref, i) => (
                                    <li key={i} style={{ marginBottom: 4 }}>{ref}</li>
                                ))}
                            </ul>
                        ) : (
                            <p style={{ color: 'var(--text-muted)' }}>No references yet.</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

/* ── Helpers ─────────────────────────────────────────────── */

/** Safely convert any value to a string for rendering */
function safe(v: unknown): string {
    if (v == null) return '';
    if (typeof v === 'string') return v;
    if (typeof v === 'number') return String(v);
    if (typeof v === 'boolean') return v ? 'Yes' : 'No';
    try { return JSON.stringify(v); } catch { return String(v); }
}

function formatDist(v: unknown): string {
    const n = Number(v);
    if (isNaN(n)) return safe(v) + ' m';
    if (n >= 1000) return `${(n / 1000).toFixed(1)} km`;
    return `${n.toFixed(0)} m`;
}

function InfoRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
    return (
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '4px 0' }}>
            <span style={{ color: 'var(--accent)', marginTop: 2, opacity: 0.9 }}>{icon}</span>
            <div>
                <div style={{ fontWeight: 500, fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.02em', marginBottom: 2 }}>{label}</div>
                <div style={{ color: 'var(--text-primary)', fontSize: '0.85rem', lineHeight: 1.4, fontWeight: 400 }}>{value}</div>
            </div>
        </div>
    );
}

function downloadText(content: string) {
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const blob = new Blob([content], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `climsight_report_${ts}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
}

function downloadMarkdown(content: string) {
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const blob = new Blob([content], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `climsight_report_${ts}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
}

/* ── Categorized Plots ────────────────────────────────────── */

function categorizePlot(url: string): string {
    const name = url.split('/').pop()?.toLowerCase() ?? '';
    if (/climate|temperature|precipitation|wind/.test(name)) return 'Climate Data';
    if (/disaster|hazard/.test(name)) return 'Natural Hazards';
    if (/population/.test(name)) return 'Population Data';
    return 'Additional Analysis';
}

function CategorizedPlots({ plots }: { plots: string[] }) {
    const groups: Record<string, string[]> = {};
    for (const url of plots) {
        const cat = categorizePlot(url);
        (groups[cat] ??= []).push(url);
    }
    const order = ['Climate Data', 'Natural Hazards', 'Population Data', 'Additional Analysis'];
    return (
        <>
            {order.filter((c) => groups[c]).map((category) => (
                <div key={category} style={{ marginTop: 12 }}>
                    <h4 style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
                        {category}
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {groups[category].map((url, i) => (
                            <img
                                key={i}
                                src={url}
                                alt={`${category} ${i + 1}`}
                                style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }}
                            />
                        ))}
                    </div>
                </div>
            ))}
        </>
    );
}

/* ── Data Tab ─────────────────────────────────────────────── */

function DataTab({ datasets, sessionId }: { datasets: Array<{ path: string; label: string; source: string }>; sessionId?: string }) {
    if (datasets.length === 0) {
        return (
            <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                <Database size={28} style={{ opacity: 0.3, marginBottom: 8 }} />
                <p>No datasets were generated for this query.</p>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: '0.85rem' }}>
            <h3 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)', borderBottom: '1px solid var(--border)', paddingBottom: 6 }}>
                Available Datasets
            </h3>
            {datasets.map((ds, idx) => {
                const filename = ds.path.split('/').pop() ?? 'dataset';
                // Extract relative path from sandbox root for the download endpoint.
                // Paths look like /tmp/sandbox/<uuid>/climate_data/file.csv
                // We need: climate_data/file.csv (relative to uuid_main_dir)
                const sandboxParts = ds.path.split('/sandbox/');
                let relativePath = filename;  // fallback: just the basename
                if (sandboxParts.length > 1) {
                    // After '/sandbox/' we have '<uuid>/sub/path/file.csv'
                    // Strip the UUID segment
                    const afterSandbox = sandboxParts[1];
                    const slashIdx = afterSandbox.indexOf('/');
                    if (slashIdx !== -1) {
                        relativePath = afterSandbox.substring(slashIdx + 1);
                    }
                }
                const downloadUrl = sessionId
                    ? `/api/datasets/${sessionId}/${relativePath}`
                    : `/artifacts/${filename}`;
                return (
                    <div
                        key={idx}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '10px 14px',
                            borderRadius: 8,
                            border: '1px solid var(--border)',
                            background: 'var(--bg-secondary)',
                        }}
                    >
                        <div>
                            <div style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{ds.label}</div>
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 2 }}>
                                {ds.source} — {filename}
                            </div>
                        </div>
                        <a
                            href={downloadUrl}
                            download={filename}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 4,
                                padding: '6px 12px',
                                borderRadius: 6,
                                border: '1px solid var(--accent)',
                                color: 'var(--accent)',
                                fontSize: '0.78rem',
                                textDecoration: 'none',
                                transition: 'all 0.2s',
                            }}
                        >
                            <Download size={13} />
                            Download
                        </a>
                    </div>
                );
            })}
        </div>
    );
}
