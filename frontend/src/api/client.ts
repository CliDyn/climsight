/**
 * ClimSight API Client — REST + WebSocket
 */
import axios, { type AxiosInstance } from 'axios';

const BASE = '';  // proxied by Vite in dev

// ── REST Client ────────────────────────────────────────── //

const http: AxiosInstance = axios.create({ baseURL: BASE });

export interface Session {
    session_id: string;
    created_at: string;
    config: Record<string, unknown>;
}

export interface ModelInfo {
    models: Record<string, { model_type: string; model_name: string }>;
}

export interface ClimateSource {
    name: string;
    enabled: boolean;
    description: string;
}

// Sessions
export const createSession = () => http.post<Session>('/api/sessions');
export const getSession = (id: string) => http.get<Session>(`/api/sessions/${id}`);
export const deleteSession = (id: string) => http.delete(`/api/sessions/${id}`);

// Config
export const getModels = () => http.get<ModelInfo>('/api/sessions/models');
export const getClimSources = () => http.get<{ sources: ClimateSource[] }>('/api/sessions/climate-sources');
export const selectModel = (sid: string, slot: string, model: string) =>
    http.put(`/api/sessions/${sid}/model`, { slot, model_name: model });

// Health
export const healthCheck = () => http.get('/health');

// ── WebSocket Client ───────────────────────────────────── //

export type WsMessageType = 'status' | 'location' | 'response' | 'error' | 'complete' | 'plot';

export interface WsMessage {
    type: WsMessageType;
    data: unknown;
}

export interface AnalysisRequest {
    lat: number;
    lon: number;
    query: string;
    config?: Record<string, unknown>;
}

export function connectAnalysis(
    sessionId: string,
    request: AnalysisRequest,
    onMessage: (msg: WsMessage) => void,
    onClose?: () => void,
    onError?: (err: Event) => void,
): WebSocket {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = window.location.host;
    const ws = new WebSocket(`${proto}://${host}/api/sessions/${sessionId}/agent/ws`);

    ws.onopen = () => {
        ws.send(JSON.stringify(request));
    };

    ws.onmessage = (event) => {
        try {
            const msg: WsMessage = JSON.parse(event.data);
            onMessage(msg);
        } catch {
            console.warn('Unparseable WS message', event.data);
        }
    };

    ws.onclose = () => onClose?.();
    ws.onerror = (e) => onError?.(e);

    return ws;
}
