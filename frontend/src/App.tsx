import { useState, useCallback, useEffect, useRef } from 'react';
import { MapPanel } from './components/MapPanel';
import { QueryForm } from './components/QueryForm';
import { ReportView } from './components/ReportView';
import { StatusBar } from './components/StatusBar';
import { SettingsPanel, type AnalysisConfig } from './components/SettingsPanel';
import {
  createSession,
  connectAnalysis,
  type Session,
  type WsMessage,
} from './api/client';
import { Compass, Sun, Moon } from 'lucide-react';
import './index.css';

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

interface AnalysisState {
  running: boolean;
  status: string;
  location: LocationData | null;
  report: string;
  plots: string[];
  error: string | null;
  inputParams: Record<string, unknown> | null;
  references: { used?: string[] } | null;
}

const INITIAL_ANALYSIS: AnalysisState = {
  running: false,
  status: '',
  location: null,
  report: '',
  plots: [],
  error: null,
  inputParams: null,
  references: null,
};

const DEFAULT_CONFIG: AnalysisConfig = {
  model_name: 'gpt-5-mini',
  climate_data_source: 'nextGEMS',
  use_smart_agent: false,
  use_era5_data: false,
  use_powerful_data_analysis: false,
};

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [coords, setCoords] = useState<[number, number]>([52.52, 13.405]);
  const [question, setQuestion] = useState('');
  const [analysis, setAnalysis] = useState<AnalysisState>(INITIAL_ANALYSIS);
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    return (localStorage.getItem('theme') as 'dark' | 'light') || 'dark';
  });
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    createSession().then((res) => setSession(res.data));
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  const handleMapClick = useCallback((lat: number, lng: number) => {
    setCoords([lat, lng]);
  }, []);

  const handleSubmit = useCallback(() => {
    if (!session) return;
    if (wsRef.current) wsRef.current.close();

    setAnalysis({ ...INITIAL_ANALYSIS, running: true, status: 'Connecting...' });

    // Build the payload with config overrides
    const payload = {
      lat: coords[0],
      lon: coords[1],
      query: question,
      config: {
        model_name: config.model_name,
        climate_data_source: config.climate_data_source,
        use_smart_agent: config.use_smart_agent,
        use_era5_data: config.use_era5_data,
        use_powerful_data_analysis: config.use_powerful_data_analysis,
      },
    };

    wsRef.current = connectAnalysis(
      session.session_id,
      payload,
      (msg: WsMessage) => {
        const d = msg.data as Record<string, unknown>;
        setAnalysis((prev) => {
          switch (msg.type) {
            case 'status':
              return { ...prev, status: (d?.message ?? d) as string };
            case 'location':
              return { ...prev, location: d as unknown as LocationData };
            case 'response': {
              const content = (d?.content ?? d) as string;
              const plotUrls = (d?.plot_urls ?? []) as string[];
              const params = (d?.input_params ?? null) as Record<string, unknown> | null;
              const refs = (d?.references ?? null) as { used?: string[] } | null;
              return {
                ...prev,
                report: content,
                plots: [...prev.plots, ...plotUrls],
                inputParams: params,
                references: refs,
                running: false,
                status: 'Analysis complete',
              };
            }
            case 'plot':
              return { ...prev, plots: [...prev.plots, String(d)] };
            case 'error':
              return { ...prev, error: (d?.message ?? JSON.stringify(d)) as string, running: false };
            case 'complete':
              return { ...prev, running: false, status: 'Analysis complete' };
            default:
              return prev;
          }
        });
      },
      () => setAnalysis((prev) => ({ ...prev, running: false })),
      () => setAnalysis((prev) => ({ ...prev, error: 'WebSocket error', running: false })),
    );
  }, [session, coords, question, config]);

  return (
    <div className="app-shell">
      {/* ── Top bar ──────────────────────────── */}
      <header className="top-bar">
        <div className="flex items-center gap-3">
          <div className="icon-box">
            <Compass size={18} className="text-sky-400" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight leading-none">ClimSight</h1>
            <p className="text-[11px] leading-none mt-0.5" style={{ color: 'var(--text-muted)' }}>
              Climate Intelligence Platform
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button className="theme-toggle" onClick={toggleTheme} title="Toggle theme">
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {session ? `Session ${session.session_id.slice(0, 8)}…` : 'Connecting…'}
          </div>
        </div>
      </header>

      {/* ── Main split: map left, content right ─ */}
      <div className="split-layout">
        <div className="map-pane">
          <MapPanel lat={coords[0]} lng={coords[1]} onClick={handleMapClick} theme={theme} />
        </div>

        <div className="content-pane">
          {/* Settings */}
          <SettingsPanel config={config} onChange={setConfig} disabled={analysis.running} />

          {/* Query form */}
          <QueryForm
            lat={coords[0]}
            lng={coords[1]}
            question={question}
            onQuestionChange={setQuestion}
            onSubmit={handleSubmit}
            disabled={analysis.running || !session}
          />

          {/* Status / Error */}
          {(analysis.running || analysis.status || analysis.error) && (
            <StatusBar
              status={analysis.status}
              running={analysis.running}
              error={analysis.error}
            />
          )}

          {/* Report */}
          <ReportView
            report={analysis.report}
            plots={analysis.plots}
            location={analysis.location}
            running={analysis.running}
            inputParams={analysis.inputParams ?? undefined}
            references={analysis.references ?? undefined}
          />
        </div>
      </div>
    </div>
  );
}
