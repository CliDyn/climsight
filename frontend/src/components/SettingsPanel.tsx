import { useState } from 'react';
import { ChevronDown, ChevronUp, Cpu, Database, Search, Code, Zap } from 'lucide-react';

const MODEL_OPTIONS = [
    'gpt-5.2',
    'gpt-5.2-mini',
    'gpt-5-nano',
    'gpt-5-mini',
    'gpt-5',
    'gpt-4.1-nano',
    'gpt-4.1-mini',
    'gpt-4.1',
];

const CLIMATE_SOURCES = [
    { value: 'nextGEMS', label: 'nextGEMS (High resolution)' },
    { value: 'ICCP', label: 'ICCP (AWI-CM3, medium resolution)' },
    { value: 'AWI_CM', label: 'AWI-CM (CMIP6, low resolution)' },
    { value: 'DestinE', label: 'DestinE IFS-FESOM (High resolution, SSP3-7.0)' },
];

export interface AnalysisConfig {
    model_name: string;
    climate_data_source: string;
    use_smart_agent: boolean;
    use_era5_data: boolean;
    use_powerful_data_analysis: boolean;
}

interface Props {
    config: AnalysisConfig;
    onChange: (config: AnalysisConfig) => void;
    disabled?: boolean;
}

export function SettingsPanel({ config, onChange, disabled }: Props) {
    const [expanded, setExpanded] = useState(true);

    const set = <K extends keyof AnalysisConfig>(key: K, val: AnalysisConfig[K]) =>
        onChange({ ...config, [key]: val });

    return (
        <div className="glass" style={{ overflow: 'hidden' }}>
            {/* Header — always visible */}
            <button
                className="settings-header"
                onClick={() => setExpanded((e) => !e)}
                style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '10px 14px',
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-primary)',
                    cursor: 'pointer',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                }}
            >
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Zap size={14} style={{ color: 'var(--accent)' }} />
                    Settings
                </span>
                {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>

            {expanded && (
                <div style={{ padding: '0 14px 14px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {/* Model selector */}
                    <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                            <Cpu size={12} /> Model for synthesis
                        </span>
                        <select
                            className="input"
                            value={config.model_name}
                            onChange={(e) => set('model_name', e.target.value)}
                            disabled={disabled}
                            style={{ fontSize: '0.8rem' }}
                        >
                            {MODEL_OPTIONS.map((m) => (
                                <option key={m} value={m}>{m}</option>
                            ))}
                        </select>
                    </label>

                    {/* Climate data source */}
                    <label style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}>
                            <Database size={12} /> Climate data source
                        </span>
                        <select
                            className="input"
                            value={config.climate_data_source}
                            onChange={(e) => set('climate_data_source', e.target.value)}
                            disabled={disabled}
                            style={{ fontSize: '0.8rem' }}
                        >
                            {CLIMATE_SOURCES.map((s) => (
                                <option key={s.value} value={s.value}>{s.label}</option>
                            ))}
                        </select>
                    </label>

                    {/* Toggles row */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        <Toggle
                            icon={<Search size={12} />}
                            label="Extra search"
                            help="Additional Wikipedia & RAG requests (slower)"
                            checked={config.use_smart_agent}
                            onChange={(v) => set('use_smart_agent', v)}
                            disabled={disabled}
                        />
                        <Toggle
                            icon={<Database size={12} />}
                            label="ERA5 data"
                            help="Retrieve ERA5 time series from Arraylake"
                            checked={config.use_era5_data}
                            onChange={(v) => set('use_era5_data', v)}
                            disabled={disabled}
                        />
                        <Toggle
                            icon={<Code size={12} />}
                            label="Python analysis"
                            help="Allow Python REPL & plot generation"
                            checked={config.use_powerful_data_analysis}
                            onChange={(v) => set('use_powerful_data_analysis', v)}
                            disabled={disabled}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}

/* -- Toggle chip --------------------------------------------------------- */
function Toggle({
    icon,
    label,
    help,
    checked,
    onChange,
    disabled,
}: {
    icon: React.ReactNode;
    label: string;
    help: string;
    checked: boolean;
    onChange: (v: boolean) => void;
    disabled?: boolean;
}) {
    return (
        <button
            onClick={() => !disabled && onChange(!checked)}
            title={help}
            style={{
                display: 'flex',
                alignItems: 'center',
                gap: 5,
                padding: '5px 10px',
                borderRadius: 6,
                border: `1px solid ${checked ? 'var(--accent)' : 'var(--border)'}`,
                background: checked ? 'rgba(56, 189, 248, 0.12)' : 'transparent',
                color: checked ? 'var(--accent)' : 'var(--text-secondary)',
                cursor: disabled ? 'not-allowed' : 'pointer',
                fontSize: '0.75rem',
                fontFamily: 'var(--font)',
                opacity: disabled ? 0.5 : 1,
                transition: 'all 0.2s',
            }}
        >
            {icon}
            {label}
        </button>
    );
}
