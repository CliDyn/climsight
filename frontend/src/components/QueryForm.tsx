import { MapPin, Send } from 'lucide-react';

interface Props {
    lat: number;
    lng: number;
    question: string;
    onQuestionChange: (q: string) => void;
    onSubmit: () => void;
    disabled: boolean;
}

export function QueryForm({ lat, lng, question, onQuestionChange, onSubmit, disabled }: Props) {
    return (
        <div className="glass" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
            {/* Coordinates display */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem' }}>
                <MapPin size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                <span style={{ color: 'var(--text-secondary)' }}>
                    {lat.toFixed(4)}°, {lng.toFixed(4)}°
                </span>
            </div>

            {/* Question input */}
            <textarea
                className="input"
                rows={3}
                style={{ resize: 'none' }}
                placeholder="What do you want to know about this location? e.g. 'What are the climate risks for agriculture?'"
                value={question}
                onChange={(e) => onQuestionChange(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        if (!disabled) onSubmit();
                    }
                }}
            />

            {/* Submit */}
            <button
                className="btn-primary"
                onClick={onSubmit}
                disabled={disabled}
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
            >
                <Send size={14} />
                Analyze Location
            </button>
        </div>
    );
}
