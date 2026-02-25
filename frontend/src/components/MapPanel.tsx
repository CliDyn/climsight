import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Custom SVG marker
const pinIcon = L.divIcon({
    className: '',
    iconSize: [28, 40],
    iconAnchor: [14, 40],
    html: `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="40" viewBox="0 0 28 40" fill="none">
    <path d="M14 0C6.268 0 0 6.268 0 14c0 10.5 14 26 14 26s14-15.5 14-26C28 6.268 21.732 0 14 0z" fill="#0284c7"/>
    <circle cx="14" cy="14" r="6" fill="#fff"/>
  </svg>`,
});

const TILES = {
    dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    light: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
};

interface Props {
    lat: number;
    lng: number;
    onClick: (lat: number, lng: number) => void;
    theme: 'dark' | 'light';
}

function ClickHandler({ onClick }: { onClick: Props['onClick'] }) {
    useMapEvents({
        click(e) {
            onClick(e.latlng.lat, e.latlng.lng);
        },
    });
    return null;
}

export function MapPanel({ lat, lng, onClick, theme }: Props) {
    const mapRef = useRef<L.Map | null>(null);

    useEffect(() => {
        mapRef.current?.flyTo([lat, lng], mapRef.current.getZoom(), { animate: true, duration: 0.6 });
    }, [lat, lng]);

    return (
        <MapContainer
            center={[lat, lng]}
            zoom={6}
            ref={mapRef}
            style={{ width: '100%', height: '100%' }}
            zoomControl={false}
        >
            <TileLayer
                attribution='&copy; <a href="https://carto.com/">CARTO</a>'
                url={TILES[theme]}
                key={theme}
            />
            <Marker position={[lat, lng]} icon={pinIcon} />
            <ClickHandler onClick={onClick} />
        </MapContainer>
    );
}
