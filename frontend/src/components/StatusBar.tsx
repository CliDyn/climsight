import { AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

interface Props {
    status: string;
    running: boolean;
    error: string | null;
}

export function StatusBar({ status, running, error }: Props) {
    if (error) {
        return (
            <div className="badge badge-error" style={{ width: '100%', justifyContent: 'flex-start' }}>
                <AlertCircle size={14} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{error}</span>
            </div>
        );
    }

    if (running) {
        return (
            <div className="badge badge-info" style={{ width: '100%', justifyContent: 'flex-start' }}>
                <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{status || 'Processing...'}</span>
            </div>
        );
    }

    return (
        <div className="badge badge-success" style={{ width: '100%', justifyContent: 'flex-start' }}>
            <CheckCircle2 size={14} />
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{status}</span>
        </div>
    );
}
