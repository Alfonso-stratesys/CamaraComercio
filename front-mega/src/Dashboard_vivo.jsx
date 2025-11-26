import React, { useState, useEffect } from 'react';
import { 
  MessageSquare, 
  Users, 
  TrendingUp, 
  Activity, 
  Wifi 
} from 'lucide-react';

import { supabase } from "./supabaseClient";

// ==========================================
// CAPA DE SERVICIO REAL (SUPABASE)
// ==========================================

async function fetchLatestResponses(limit = 6) {
  const { data, error } = await supabase
    .from("respuestas")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(limit);

  if (error) {
    console.error("Error fetching respuestas:", error);
    return [];
  }

  return data.map((r) => ({
    id: r.id,
    usuario: r.industria ?? "Usuario",
    texto: r.proposito_hoy ?? "Sin texto",
    timestamp: new Date(r.created_at),
    categoria: r.adopcion_tech ?? "General"
  }));
}

async function fetchTotalUsers() {
  const { count, error } = await supabase
    .from("respuestas")
    .select("*", { count: "exact", head: true });

  if (error) {
    console.error("Error counting:", error);
    return 0;
  }

  return count;
}

async function fetchNextResponse(lastId, shownIds) {
  const { data, error } = await supabase
    .from("respuestas")
    .select("*")
    .order("created_at", { ascending: false });

  if (error || !data || data.length === 0) return null;

  // Filtramos las que ya se han mostrado
  const remaining = data.filter(row => !shownIds.includes(row.id));

  // Si quedan respuestas nuevas → toma la más reciente
  if (remaining.length > 0) return remaining[0];

  // Si ya se mostraron todas → reiniciar ciclo
  return data[0];
}



// ==========================================
// CONFIGURACIÓN Y ESTILOS
// ==========================================

const COLORS = {
  primary: '#E30613',    // Rojo Principal
  secondary: '#BA0C2F',  // Rojo Oscuro
  white: '#FFFFFF',
  gray: '#F9F9F9',
  text: '#1F2937',
  chartColors: ['#E30613', '#BA0C2F', '#FF6B6B', '#4B5563']
};

// URL para generar el QR dinámicamente
const QR_TARGET_URL = "https://tu-evento-mega.com/participar"; 
const QR_API_URL = `https://api.qrserver.com/v1/create-qr-code/?size=400x400&color=E30613&data=${encodeURIComponent(QR_TARGET_URL)}`;

// ==========================================
// CAPA DE SERVICIO (MOCK BACKEND)
// ==========================================

const MockBackendService = {
  // Generador de nombres aleatorios
  sectores: ['Ana G.', 'Carlos M.', 'Sofia L.', 'Javier R.', 'Equipo IT', 'Mkt Dept'],
  
  // Generador de frases corporativas
  frases: [
    "Necesitamos agilizar los procesos de aprobación.",
    "La integración con IA es clave para el 2025.",
    "Más espacios de colaboración remota.",
    "Excelente visión, totalmente alineados.",
    "¿Cómo impactará esto en el presupuesto Q3?",
    "Propongo revisar la infraestructura cloud."
  ],

  /**
   * Simula obtener una nueva respuesta de la base de datos.
   */
  fetchLatestResponse: () => {
    return new Promise((resolve) => {
      setTimeout(() => {
        const randomName = MockBackendService.sectores[Math.floor(Math.random() * MockBackendService.sectores.length)];
        const randomText = MockBackendService.frases[Math.floor(Math.random() * MockBackendService.frases.length)];
        const categories = ['Innovación', 'Procesos', 'Cultura', 'General'];
        
        resolve({
          id: Date.now().toString(),
          usuario: randomName,
          texto: randomText,
          timestamp: new Date(),
          categoria: categories[Math.floor(Math.random() * categories.length)]
        });
      }, 300);
    });
  },

  /**
   * Simula obtener métricas actualizadas
   */
  fetchMetrics: (baseTotal) => {
    return new Promise((resolve) => {
      resolve({
        totalParticipantes: baseTotal + Math.floor(Math.random() * 3),
        tasaAdopcion: 75 + Math.floor(Math.random() * 10),
        respuestasPorMinuto: 12 + Math.floor(Math.random() * 5)
      });
    });
  }
};

// ==========================================
// COMPONENTES UI
// ==========================================

/**
 * Tarjeta simple para KPIs numéricos.
 */
const KPICard = ({ title, value, icon: Icon, subtext }) => (
  <div className="bg-white p-6 rounded-xl shadow-sm border-l-4 border-[color:var(--primary-color)] flex items-center justify-between transition-transform hover:scale-105 duration-300">
    <div>
      <p className="text-gray-500 text-sm font-medium uppercase tracking-wider">{title}</p>
      <h3 className="text-3xl font-bold text-gray-800 mt-1">{value}</h3>
    </div>
    <div className="bg-red-50 p-3 rounded-full">
      <Icon size={24} color={COLORS.primary} />
    </div>
  </div>
);

const BigKPI = ({ title, value, icon: Icon }) => (
  <div className="h-full w-full flex flex-col items-center justify-center p-12">

    <Icon size={60} className="text-[#E30613] mb-6 opacity-80" />

    <h3 className="text-3xl font-semibold text-gray-600 tracking-wide mb-4">
      {title}
    </h3>

    <p className="text-[7rem] font-extrabold text-gray-900 leading-none">
      {value}
    </p>

  </div>
);


/**
 * Componente de lista de feed con animación.
 */
const FeedItem = ({ item }) => (
   <div className={`${item.isNew ? "animate-new" : "animate-slideIn"} mb-4 p-4 bg-white rounded-lg shadow-sm border border-gray-100 hover:border-red-100 transition-colors`}>
    <div className="flex justify-between items-start mb-2">
     <span className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>

    <span className="text-gray-500 text-sm">
        Sector:
    </span>

    <span className="text-gray-800 font-semibold capitalize">
        {item.usuario}
    </span>
</span>


      <span className="text-xs text-gray-400 bg-gray-100 px-2 py-1 rounded-full">
        {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </span>
    </div>
    <p className="text-gray-600 text-sm leading-relaxed">
      "{item.texto}"
    </p>
   <div className="mt-3 flex justify-end">
  <span className="text-xs font-semibold px-3 py-1 rounded-full bg-red-100 text-red-700 border border-red-200">
    Adopción tecnológica:{" "}
    <span className="capitalize font-bold">{item.categoria}</span>
  </span>
</div>

  </div>
);

// ==========================================
// COMPONENTE PRINCIPAL (DASHBOARD)
// ==========================================

export default function DashboardPresentation() {
  // -- ESTADOS --
  const [respuestas, setRespuestas] = useState([]);
  const [metrics, setMetrics] = useState({
    totalParticipantes: 124,
    tasaAdopcion: 78,
    respuestasPorMinuto: 15
  });

   const [lastSeenId, setLastSeenId] = useState(null);
   const [shownIds, setShownIds] = useState([]);
   const [pendingQueue, setPendingQueue] = useState([]);

  // -- EFECTOS --

 useEffect(() => {
  async function loadInitial() {
    const initial = await fetchLatestResponses(6);

    setRespuestas(initial);
    setShownIds(initial.map(r => r.id));

    if (initial.length > 0) {
      setLastSeenId(initial[0].id);
    }

    const total = await fetchTotalUsers();
    setMetrics(prev => ({
      ...prev,
      totalParticipantes: total
    }));
  }

  loadInitial();

  const intervalId = setInterval(async () => {

  // 1. Obtener TODAS las respuestas de Supabase
  const { data, error } = await supabase
    .from("respuestas")
    .select("*")
    .order("created_at", { ascending: false });

  if (error || !data) return;

  // 2. Detectar NUEVAS respuestas → No mostradas aún
  const nuevas = data.filter(r => !shownIds.includes(r.id));

  // 3. Si hay nuevas → añadirlas a la cola
  if (nuevas.length > 0) {
    setPendingQueue(prev => [...prev, ...nuevas]);
  }

  // 4. Si la cola tiene elementos → mostrar SOLO 1
  setPendingQueue(prev => {
    if (prev.length === 0) return prev;

    const next = prev[0];

    const formatted = {
      id: next.id,
      usuario: next.industria ?? "Usuario",
      texto: next.proposito_hoy ?? "Sin texto",
      timestamp: new Date(next.created_at),
      categoria: next.adopcion_tech ?? "General",
      isNew: true
    };

    // Insertar al inicio y limitar a solo 3 visibles
    setRespuestas(r => [formatted, ...r].slice(0, 3));

    // Registrar como mostrada
    setShownIds(ids => [...ids, next.id]);

    // Eliminar ese item de la cola
    return prev.slice(1);
  });

}, 4000); // ← aquí eliges el intervalo (4s)


  return () => clearInterval(intervalId);
}, [lastSeenId, shownIds]);


  // -- RENDERIZADO --

  return (
    <div className="h-screen overflow-hidden bg-gray-50 text-gray-800 font-sans flex flex-col">

      
      {/* Estilos inline para animaciones personalizadas */}
      <style>{`
        :root {
          --primary-color: ${COLORS.primary};
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slideIn {
          animation: slideIn 0.5s ease-out forwards;
        }
        /* NUEVA ANIMACIÓN */
        @keyframes slideInNew {
        0% { opacity: 0; transform: translateY(-25px); }
        100% { opacity: 1; transform: translateY(0); }
        }
        .animate-new {
            animation: slideInNew 0.6s ease-out forwards;
        }
      `}</style>

      
      {/* HEADER SUPERIOR */}
<header className="bg-white border-b border-gray-200 h-16 flex items-center justify-between px-6 shadow-sm z-10">

  {/* Botón izquierda → Volver a /Admin */}
  <button 
    onClick={() => window.location.href = "/Admin"}
    className="flex items-center gap-2 text-gray-700 hover:text-[#E30613] transition font-semibold"
  >
    ← Volver
  </button>

  {/* Logos centrados */}
  <div className="flex items-center gap-4">
    <img 
      src="/Cámara_de_Comercio_de_Bogotá_logo.png"
      alt="Cámara de Comercio"
      className="h-10 object-contain"
    />
    <img 
      src="/Stratesys.png"
      alt="Stratesys"
      className="h-7 object-contain"
    />
  </div>

  {/* Botón En vivo */}
  <button 
    className="flex items-center gap-2 bg-green-50 text-green-700 border border-green-200 px-3 py-1 rounded-full text-sm font-semibold"
  >
    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
    En vivo
  </button>

</header>




      {/* CONTENIDO PRINCIPAL */}
      <main className="flex flex-1 h-full overflow-hidden">
        
        {/* ZONA IZQUIERDA: QR */}
        <section className="w-[30%] h-[calc(100vh-64px)] overflow-hidden flex flex-col items-center justify-start bg-white p-6">
            <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-[#E30613] to-[#BA0C2F]"></div>
            
            <div className="mb-8">
                <h2 className="text-4xl font-extrabold text-gray-900 mb-2 tracking-tight">¡Participa ahora!<br/></h2>
                <p className="text-gray-500 text-lg">Rellena el formulario para participar.</p>
            </div>

            <div className="relative group">
                <div className="absolute -inset-1 bg-gradient-to-br from-[#E30613] to-pink-600 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000"></div>
                <div className="relative bg-white p-4 rounded-xl border border-gray-100 shadow-xl">
                    <img 
                        src={QR_API_URL} 
                        className="w-49 h-49 object-contain"
                    />
                </div>
            </div>

        </section>

        {/* ZONA DERECHA: DATOS */}
        <section className="w-[70%] bg-white p-8 grid grid-cols-2 gap-8 h-full">
            
            {/* COLUMNA 1: FEED */}
            <div className="flex flex-col h-full">
                <div className="flex justify-between items-center mb-6">
                    <h3 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                        <MessageSquare className="text-[#E30613]" size={24} />
                        Últimas Aportaciones
                    </h3>
                    {/*
                    <span className="text-xs font-mono bg-gray-200 text-gray-600 px-2 py-1 rounded">
                        SYNC: 4s
                    </span>
                    */}
                </div>
                
                <div className="h-[80vh] overflow-hidden pr-2 flex flex-col gap-4">

                    {respuestas.map((resp) => (
                        <FeedItem key={resp.id} item={resp} />
                    ))}
                    {respuestas.length === 0 && (
                        <div className="text-center text-gray-400 mt-20 italic">
                            Esperando primeras respuestas...
                        </div>
                    )}
                </div>
            </div>

            {/* COLUMNA 2: KPIs */}
            <div className="h-full">
                <BigKPI 
                    title="Participantes Totales"
                    value={metrics.totalParticipantes}
                    icon={Users}
                />
            </div>

        </section>
      </main>
    </div>
  );
}