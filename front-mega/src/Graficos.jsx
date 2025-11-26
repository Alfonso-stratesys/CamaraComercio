import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, Treemap
} from 'recharts';
import {
  ChevronRight, ChevronLeft, Activity, Brain, Users, Building2,
  Zap, Database, Globe, Share2, Rocket, AlertTriangle, Clock, TrendingUp
} from 'lucide-react';

// La librería @supabase/supabase-js ha sido eliminada. Usaremos 'fetch' directamente.
 
 
// --- SUPABASE INITIALIZATION (Direct Access via Env Vars) ---
// Accedemos a las variables de entorno inyectadas por el entorno de desarrollo.
// Usamos window.env para compatibilidad y evitamos 'import.meta.env' para solucionar la advertencia.
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || "";
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_KEY || "";
const TABLE_NAME = "respuestas"; // Nombre de la tabla
const BASE_URL = SUPABASE_URL.endsWith('/') ? SUPABASE_URL.slice(0, -1) : SUPABASE_URL;
const ENDPOINT = `${SUPABASE_URL}/rest/v1/${TABLE_NAME}`;
 
console.log("🔍 [ENV] VITE_SUPABASE_URL:", SUPABASE_URL);
console.log("🔍 [ENV] VITE_SUPABASE_KEY (primeros 6 chars):", SUPABASE_KEY?.substring(0, 6));
 
console.log("🔍 [ENV] ENDPOINT construido:", ENDPOINT);
console.log("🔍 [ENV] URL final usada en la llamada:", `${ENDPOINT}?select=*`);
 
 
if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.warn("⚠️ Faltan VITE_SUPABASE_URL o VITE_SUPABASE_KEY en el archivo .env. No se podrá conectar a Supabase.");
} else {
    // LOGS: Añadimos logs de depuración aquí para ver a dónde intenta conectar.
    console.log(`[DEBUG] Intentando conectar a: ${ENDPOINT}`);
    console.log(`[DEBUG] Con Supabase Key (Prefix): ${SUPABASE_KEY.substring(0, 4)}...`);
}
 
// --- CONSTANTS & PALETTE ---
const PALETTE = {
  primary: '#E30613',    // Rojo CCB
  secondary: '#BA0C2F',  // Rojo Oscuro
  dark: '#1A1A1A',
  light: '#F9F9F9',
  white: '#FFFFFF',
  text: '#333333',
  gray: '#CCCCCC',
  industries: {
    'Manufactura': '#F46A6A',
    'Construcción': '#F4B6C2',
    'Tecnología': '#5CA6D1',
    'Comercio': '#F4C542',
    'Energía y Minería': '#C7A8EA',
    'Agroindustria': '#9DD9C4',
    'Salud': '#30A66D',
    'Servicios': '#5BB3C4',
    'Otra': '#EFA86E'
  },
  levels: {
    'Bajo': '#E30613',
    'Medio': '#F4C542',
    'Alto': '#30A66D',
    'Avanzado': '#0E3A63'
  }
};
// Orden fijo para las categorías de empleados (leyenda y donut)
const EMPLOYEE_ORDER = ["1-50", "51-200", "201-500", ">500"];

// Colores fijos por categoría de empleados (SIN tipos TS)
const EMPLOYEE_COLORS = {
  "1-50": "#E30613",   // rojo
  "51-200": "#0E3A63", // azul oscuro
  "201-500": "#5CA6D1",// azul claro
  ">500": "#F4C542",   // amarillo
};
 
// --- DATA SERVICE (SUPABASE ONLY) ---
const DataService = {
  /**
   * Intenta obtener datos reales de Supabase usando la API REST nativa.
   * Devuelve un array con los datos o null/empty array si hay error/no hay datos.
   */
  fetchFromSupabase: async () => {
    if (!SUPABASE_URL || !SUPABASE_KEY) {
        console.error("Faltan credenciales de Supabase. No se puede conectar.");
        return null;
    }
 
    const headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": `Bearer ${SUPABASE_KEY}`,
        "Accept": "application/json",
    };
 
    try {
        const response = await fetch(`${ENDPOINT}?select=*`, { headers });
       
        console.log("📡 [RESPONSE] Status:", response.status);
        console.log("📡 [RESPONSE] Status Text:", response.statusText);
 
        const raw = await response.clone().text();
        console.log("📡 [RESPONSE] Raw Body:", raw);
 
        if (!response.ok) {
            throw new Error(`Error HTTP: ${response.status}`);
        }
       
        const data = await response.json();
        console.log("📡 [RESPONSE] Parsed JSON Data:", data);
        // Si no hay datos, devolvemos un array vacío para indicar que la consulta fue exitosa pero no hay registros
        return data || [];
    } catch (err) {
        console.error("Error conectando a Supabase REST API:", err.message);
        // Si hay un error de conexión, devolvemos null para indicar el fallo
        return null;
    }
  },
 
    getDistributionData: async () => {
    const realData = await DataService.fetchFromSupabase();

    if (realData === null) {
      return { error: "No se pudo conectar a la base de datos.", treemap: [], employees: { total: [] } };
    }

    if (realData.length === 0) {
      return { empty: "No se encontraron datos en la tabla.", treemap: [], employees: { total: [] } };
    }

    console.log("📡 [Service] Procesando datos...");

    // --- Estructuras de acumulación ---
    const sectors = {}; // Empresas por industria
    const employeesGroups = {
      "1-50": 0,
      "51-200": 0,
      "201-500": 0,
      ">500": 0,
    };
    const sectorsTech = {}; // Adopción por industria
    const salesVol = {};    // Adopción por volumen de ventas

    // --- Diccionario de industrias (sin tildes -> canonical) ---
    const INDUSTRY_MAP = {
      agroindustria: "Agroindustria",
      manufactura: "Manufactura",
      comercio: "Comercio",
      tecnologia: "Tecnología",
      construccion: "Construcción",
      "energia y mineria": "Energía y Minería",
      servicios: "Servicios",
      servicio: "Servicios",
      salud: "Salud",
      otra: "Otra",
      otros: "Otra",
    };

    // --- Claves canónicas de adopción tecnológica ---
    const ADOPTION_KEYS = {
      BAJO: "Bajo - Uso limitado de herramientas tecnologicas basicas",
      MEDIO: "Medio - Digitalizacion de algunos procesos",
      ALTO: "Alto - Automatizacion, analitica, plataformas integradas",
      AVANZADO: "Avanzado - Uso intensivo de tecnologias emergentes, IA, IoT, etc.",
    };

    const ALL_ADOPTION_KEYS = [
      ADOPTION_KEYS.BAJO,
      ADOPTION_KEYS.MEDIO,
      ADOPTION_KEYS.ALTO,
      ADOPTION_KEYS.AVANZADO,
    ];

    const initAdoptionObject = () => ({
      [ADOPTION_KEYS.BAJO]: 0,
      [ADOPTION_KEYS.MEDIO]: 0,
      [ADOPTION_KEYS.ALTO]: 0,
      [ADOPTION_KEYS.AVANZADO]: 0,
    });

    // --- Loop principal sobre las filas reales ---
    realData.forEach((row) => {
      // 1) INDUSTRIA
      const industriaRaw = row.industria
        ? row.industria.normalize("NFD").replace(/[\u0300-\u036f]/g, "").trim().toLowerCase()
        : "otra";

      const industria = INDUSTRY_MAP[industriaRaw] || "Otra";

      // Contar empresas por industria
      sectors[industria] = (sectors[industria] || 0) + 1;

      // Asegurar estructura de adopción por industria
      if (!sectorsTech[industria]) {
        sectorsTech[industria] = initAdoptionObject();
      }

      // Normalizar guiones: convertir EN DASH (–), EM DASH (—) u otros a "-" estándar
      let numEmpleados = (row.empleados || "1-50")
        .replace(/[\u2013\u2014\u2212]/g, "-")   // remplaza – — − por -
        .trim();

      // Si no está en el diccionario por orden, se cae a ">500" o el default
      if (!EMPLOYEE_ORDER.includes(numEmpleados)) {
        numEmpleados = ">500"; 
      }

      // Acumular
      employeesGroups[numEmpleados] = (employeesGroups[numEmpleados] || 0) + 1;


      // 3) ADOPCIÓN TECNOLÓGICA (misma etiqueta que metes en la BD)
    // Normalizar tildes para comparar correctamente
    let adopcion = (row.adopcion_tech || ADOPTION_KEYS.BAJO)
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .trim();

    // Normalizar también las claves para comparar sin tildes
    const normalizedKeys = ALL_ADOPTION_KEYS.map(k =>
      k.normalize("NFD").replace(/[\u0300-\u036f]/g, "")
    );

    // Buscar índice de la clave canonizada
    const idx = normalizedKeys.indexOf(adopcion);

    if (idx >= 0) {
      adopcion = ALL_ADOPTION_KEYS[idx]; // recupera versión original correcta
    } else {
      adopcion = ADOPTION_KEYS.BAJO; // fallback
    }

    sectorsTech[industria][adopcion] += 1;


      // 4) VOLUMEN DE VENTAS
      const ventas =
        row.volumen_ventas && row.volumen_ventas.trim() !== ""
          ? row.volumen_ventas.trim()
          : "Otros";

      if (!salesVol[ventas]) {
        salesVol[ventas] = initAdoptionObject();
      }

    let adopcionSales = (row.adopcion_tech || ADOPTION_KEYS.BAJO)
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .trim();

    const salesIdx = normalizedKeys.indexOf(adopcionSales);
    adopcionSales = salesIdx >= 0 ? ALL_ADOPTION_KEYS[salesIdx] : ADOPTION_KEYS.BAJO;

    salesVol[ventas][adopcionSales] += 1;

    });

    // --- Treemap por industria ---
  let treemapData = Object.keys(sectors).map((key) => ({
    name: key,
    size: sectors[key],
    fill: PALETTE.industries[key] || PALETTE.industries["Otra"],
  }));

  // ORDENAR POR TAMAÑO (MAYOR → MENOR)
  treemapData.sort((a, b) => b.size - a.size);

    // --- Distribución por empleados ---
    const employeeData = EMPLOYEE_ORDER.map((key) => ({
      name: key,
      value: employeesGroups[key] || 0,
    }));

    // --- Adopción tecnológica por industria (porcentajes) ---
    let techAdoptionData = Object.entries(sectorsTech).map(([industry, levels]) => {
      const total = Object.values(levels).reduce((sum, n) => sum + n, 0);
      return {
        name: industry,
        Bajo: total ? (levels[ADOPTION_KEYS.BAJO] / total) * 100 : 0,
        Medio: total ? (levels[ADOPTION_KEYS.MEDIO] / total) * 100 : 0,
        Alto: total ? (levels[ADOPTION_KEYS.ALTO] / total) * 100 : 0,
        Avanzado: total ? (levels[ADOPTION_KEYS.AVANZADO] / total) * 100 : 0,
      };
    });

    // Orden alfabético con "Otra" al final
    techAdoptionData.sort((a, b) => {
      if (a.name === "Otra") return 1;
      if (b.name === "Otra") return -1;
      return a.name.localeCompare(b.name, "es", { sensitivity: "base" });
    });

    // --- Adopción por volumen de ventas (para Block3) ---
    let salesAdoptionData = Object.entries(salesVol).map(([ventas, levels]) => {
      const total = Object.values(levels).reduce((sum, n) => sum + n, 0);
      return {
        name: ventas,
        Bajo: total ? (levels[ADOPTION_KEYS.BAJO] / total) * 100 : 0,
        Medio: total ? (levels[ADOPTION_KEYS.MEDIO] / total) * 100 : 0,
        Alto: total ? (levels[ADOPTION_KEYS.ALTO] / total) * 100 : 0,
        Avanzado: total ? (levels[ADOPTION_KEYS.AVANZADO] / total) * 100 : 0,
      };
    });

    console.log("📡 [Service] Resultado final listo.", {
      treemapData,
      employeeData,
      techAdoptionData,
      salesAdoptionData,
    });

    return {
      treemap: treemapData,
      employees: { total: employeeData },
      techAdoption: { total: techAdoptionData },
      salesAdoption: { total: salesAdoptionData },
    };
  },
};
 
// --- UI COMPONENTS ---
 
const Card = ({ children, className = "" }) => (
  <div className={`bg-white rounded-xl shadow-lg p-6 border border-gray-100 hover:shadow-xl transition-shadow duration-300 ${className}`}>
    {children}
  </div>
);
 
const SectionTitle = ({ title, subtitle }) => (
  <div className="mb-6">
    <h2 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
      <div className="w-2 h-8 bg-[#E30613] rounded-full"></div>
      {title}
    </h2>
    {subtitle && <p className="text-gray-500 mt-1 ml-5 text-lg">{subtitle}</p>}
  </div>
);
 
const LoadingOverlay = ({ text }) => (
  <div className="absolute inset-0 bg-white/90 backdrop-blur-sm z-50 flex flex-col items-center justify-center animate-fadeIn">
    <div className="relative w-24 h-24 mb-6">
      <div className="absolute inset-0 border-4 border-gray-200 rounded-full"></div>
      <div className="absolute inset-0 border-4 border-[#E30613] rounded-full border-t-transparent animate-spin"></div>
      <Brain className="absolute inset-0 m-auto text-[#E30613] animate-pulse" size={32} />
    </div>
    <h3 className="text-2xl font-bold text-[#E30613] animate-pulse">{text}</h3>
    <p className="text-gray-500 mt-2">Intentando conexión con la API REST de Supabase...</p>
  </div>
);

// Nueva Animación de Carga - ESPECÍFICA PARA EL BLOQUE 6 (Síntesis de IA)
const LoadingOverlayBlock6 = () => (
    <div className="h-full w-full flex flex-col items-center justify-center bg-black text-white">
        <Brain size={64} className="text-[#E30613] animate-bounce mb-6" />
        <h2 className="text-3xl font-mono animate-pulse">GENERANDO VISIÓN CONJUNTA...</h2>
        <div className="w-64 h-2 bg-gray-800 rounded mt-4 overflow-hidden">
            <div className="h-full bg-[#E30613] animate-progress"></div>
        </div>
    </div>
);
 
const NoDataMessage = ({ message, isError }) => (
  <div className="h-full flex flex-col items-center justify-center text-center p-10 bg-gray-50 rounded-xl border-4 border-dashed border-gray-200">
    <AlertTriangle size={48} className={isError ? "text-red-500" : "text-yellow-500"} />
    <h3 className="mt-4 text-2xl font-bold text-gray-700">{isError ? "Error de Conexión" : "Sin Datos"}</h3>
    <p className="mt-2 text-gray-500">{message}</p>
    <p className="mt-4 text-sm text-gray-400">Verifica tus variables de entorno y el estado de la tabla '{TABLE_NAME}' en Supabase.</p>
  </div>
);
 
 
// --- SLIDE COMPONENTS ---
 
const IntroSlide = ({ onNext }) => (
  <div className="h-full flex flex-col items-center justify-center bg-gradient-to-br from-white to-gray-100 text-center p-10 relative overflow-hidden">
    <div className="absolute top-0 left-0 w-full h-2 bg-[#E30613]"></div>
    <div className="animate-slideUp space-y-8 z-10 max-w-4xl">
      <div className="flex justify-center items-center gap-12 mb-8 opacity-0 animate-fadeIn delay-300">

  {/* LOGO CCB */}
  <img
    src="/Cámara_de_Comercio_de_Bogotá_logo.png"
    alt="Cámara de Comercio de Bogotá"
    className="h-20 object-contain"
  />

  {/* Separador opcional */}
  <div className="w-px h-16 bg-gray-300"></div>

  {/* LOGO STRATESYS */}
  <img
    src="/Stratesys.png"
    alt="Stratesys"
    className="h-16 object-contain"
  />

</div>

      <h1 className="text-6xl font-extrabold text-gray-900 leading-tight drop-shadow-sm">
        III Encuentro Nacional de Empresas en <span className="text-[#E30613]">Trayectoria MEGA</span>
      </h1>
      <div className="pt-12">
        <button
          onClick={onNext}
          className="group bg-[#E30613] text-white px-8 py-4 rounded-full text-xl font-semibold shadow-lg hover:bg-[#BA0C2F] hover:scale-105 transition-all flex items-center gap-3 mx-auto"
        >
          Iniciar Presentación
          <ChevronRight className="group-hover:translate-x-1 transition-transform" />
        </button>
      </div>
    </div>
  </div>
);

// --- NUEVA ESTRUCTURA DE DATOS MOCK PARA IA ---
const AI_TIMELINE_DATA = [
    {
        period: "HOY (2025)", // Título fijo
        icon: Clock, // Icono fijo
        theme: "Foco en la Resiliencia",
        description: "El análisis actual se centra en la **resiliencia operativa y la digitalización básica** para asegurar la estabilidad del flujo de caja y mantener la competitividad en mercados volátiles."
    },
    {
        period: "EVOLUCIÓN", // Título fijo
        icon: TrendingUp, // Icono fijo
        theme: "La Brecha Estratégica",
        description: "La **transición** crítica exige pasar de la eficiencia (estandarización) a la creación de valor mediante la **innovación continua**, la construcción de una sólida gobernanza de datos y el desarrollo de nuevos liderazgos."
    },
    {
        period: "+18 AÑOS (2043)", // Título fijo
        icon: Rocket, // Icono fijo
        theme: "Visión de Liderazgo Global",
        description: "El futuro se define por la **integración total de la IA** para la toma de decisiones predictiva, modelos de negocio regenerativos basados en la economía circular y una fuerza laboral especializada en Deep Tech."
    }
];

 
const Block1 = ({ isActive }) => {
  const [data, setData] = useState(null);
  const [filter, setFilter] = useState(null);
  const [loading, setLoading] = useState(false);
 
  useEffect(() => {
  if (isActive && !data) {
    setLoading(true);
 
    const loadData = async () => {
      console.log("▶️ [Block1] Ejecutando getDistributionData()");
      const result = await DataService.getDistributionData();
      console.log("📊 [Block1] Resultado recibido:", result);
 
      setData(result);
      setLoading(false);
    };
 
    loadData();
  }
}, [isActive, data]);
 
  if (loading || data === null) return <LoadingOverlay text="Cargando Demografía..." />;
 
  if (data.error) {
    return <div className="p-8 h-full"><NoDataMessage message={data.error} isError={true} /></div>;
  }
 
  const totalSurveyed = data.employees.total.reduce((sum, item) => sum + item.value, 0);
 
  if (data.empty || totalSurveyed === 0) {
    return <div className="p-8 h-full"><NoDataMessage message={data.empty || "La tabla de respuestas está vacía."} isError={false} /></div>;
  }
 
  // En un escenario real, deberías tener lógica de filtrado de empleados por sector aquí.
  // Como no tenemos datos mock/real para esa complejidad, usamos el total.
// En Block1, FUERZA el orden fijo del donut:
const displayedEmployees = EMPLOYEE_ORDER.map((name) => {
  const found = data.employees.total.find((e) => e.name === name);
  return found || { name, value: 0 };
});

// 2) Para el piechart → excluir valores 0
const pieEmployees = displayedEmployees.filter((e) => e.value > 0);
 
  return (
    <div className="h-full flex flex-col p-8 animate-fadeIn">
      <SectionTitle
        title="Distribución por Industria"
        subtitle={`Total de encuestados: ${totalSurveyed}`}
      />
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-lg flex items-center gap-2">
              <Building2 size={20} className="text-[#E30613]" /> Sectores Empresariales
            </h3>
          </div>
          <div className="flex-1 relative">
             <div className="w-full h-full grid grid-cols-4 grid-rows-4 gap-1">
                {data.treemap.map((item, idx) => {
                  const spanClass =
                    idx === 0
                      ? "col-span-2 row-span-2"    // MÁS grande
                      : idx <= 2
                      ? "col-span-2 row-span-1"    // medianos
                      : "col-span-1 row-span-1";   // pequeños
                  const isSelected = filter === item.name; // Lógica de filtrado de ejemplo
                  return (
                    <div
                      key={item.name}
                      onClick={() => setFilter(item.name)}
                      className={`${spanClass} relative group cursor-pointer transition-all duration-300 overflow-hidden rounded-md border-2 ${isSelected ? 'border-black scale-[0.98]' : 'border-transparent hover:border-white hover:scale-[1.02]'}`}
                      style={{ backgroundColor: item.fill }}
                    >
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-white p-2 text-center">
                        <span className="font-bold text-shadow-sm text-sm md:text-base">{item.name}</span>
                        <span className="text-xs md:text-sm opacity-90">{item.size} empresas</span>
                      </div>
                    </div>
                  );
                })}
             </div>
          </div>
        </Card>
 
        <Card className="flex flex-col relative">
          <h3 className="font-bold text-lg mb-4 flex items-center gap-2">
            <Users size={20} className="text-[#E30613]" />
            Distribución por Número de Empleados
          </h3>
          <div className="flex-1 min-h-[300px] relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieEmployees}
                  cx="50%"
                  cy="50%"
                  innerRadius={80}
                  outerRadius={120}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name }) => `${name}`}
                  labelLine={false}
                  startAngle={90}    // empieza arriba
                  endAngle={-270}    // gira en sentido horario
                >
                  {pieEmployees.map((entry) => (
                    <Cell
                      key={`cell-${entry.name}`}
                      fill={EMPLOYEE_COLORS[entry.name]}
                    />
                  ))}
                </Pie>
                <RechartsTooltip />
                <Legend
                  verticalAlign="middle"
                  align="right"
                  layout="vertical"
                  content={() => (
                    <div
                      style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 8,
                        marginLeft: 16,
                      }}
                    >
                      {EMPLOYEE_ORDER.map((name) => (
                        <div
                          key={name}
                          style={{ display: "flex", alignItems: "center", gap: 8 }}
                        >
                          <span
                            style={{
                              width: 10,
                              height: 10,
                              borderRadius: "50%",
                              backgroundColor: EMPLOYEE_COLORS[name],
                            }}
                          />
                          <span
                            style={{
                              color: EMPLOYEE_COLORS[name],
                              fontWeight: 600,
                            }}
                          >
                            {name}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                />

              </PieChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
};
 
const Block2 = ({ isActive }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
 
  useEffect(() => {
    if (isActive && !data) {
      setLoading(true);
 
      const loadData = async () => {
        console.log("▶️ [Block2] Ejecutando getDistributionData()");
        try {
          const result = await DataService.getDistributionData(); // <-- llamada real
          console.log("📊 [Block2] Resultado recibido:", result);
          setData(result);
        } catch (error) {
          console.error("[Block2] Error al cargar datos:", error);
          setData({ error: "No se pudo cargar la información de adopción tecnológica." });
        } finally {
          setLoading(false);
        }
      };
 
      loadData();
    }
  }, [isActive, data]);
 
  if (loading || !data) return <LoadingOverlay text="Analizando Madurez Digital..." />;
 
  if (data.error) {
    return (
      <div className="p-8 h-full">
        <NoDataMessage message={data.error} isError={true} />
      </div>
    );
  }
 
  return (
    <div className="h-full flex flex-col p-8 animate-fadeIn">
      <SectionTitle title="Nivel de Adopción Tecnológica" subtitle="Madurez digital por sector industrial" />
      <Card className="flex-1 p-8">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data.techAdoption.total} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#666' }} />
            <YAxis
              unit="%"
              domain={[0, 100]}
              ticks={[0, 20, 40, 60, 80, 100]}
              tickFormatter={(value) => `${value}`}
            />
            <RechartsTooltip
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
              cursor={{ fill: 'transparent' }}
            />
            <Legend
              verticalAlign="top"
              height={70}
              content={() => (
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",   // CENTRA TODO
                    alignItems: "center",
                    gap: "35px",
                    width: "100%",
                    marginBottom: "15px",
                  }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Bajo'] }}></span>
                    Bajo
                  </span>

                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Medio'] }}></span>
                    Medio
                  </span>

                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Alto'] }}></span>
                    Alto
                  </span>

                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Avanzado'] }}></span>
                    Avanzado
                  </span>
                </div>
              )}
            />
            {['Bajo', 'Medio', 'Alto', 'Avanzado'].map((key) => (
              <Bar
                key={key}
                dataKey={key}
                name=""
                stackId="a"
                fill={PALETTE.levels[key]}
                animationDuration={1500}
                animationBegin={300}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
};
 
//----SLIDE 3-------------------------------------------------------
const Block3 = ({ isActive }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Si el slide está activo y aún no hemos cargado los datos.
    if (isActive && !data) {
      setLoading(true);

      const loadData = async () => {
        console.log("▶️ [Block3] Ejecutando getDistributionData() para Ventas/Adopción");
        try {
          // La misma función de servicio, ahora procesa los datos de ventas también
          const result = await DataService.getDistributionData();
          console.log("📊 [Block3] Resultado recibido:", result);
          setData(result);
        } catch (error) {
          console.error("[Block3] Error al cargar datos:", error);
          setData({ error: "No se pudo cargar la información de adopción por volumen de ventas." });
        } finally {
          setLoading(false);
        }
      };

      loadData();
    }
  }, [isActive, data]);
  
  // Usamos el estado de carga y error del patrón Block1/Block2
  if (loading || data === null) return <LoadingOverlay text="Correlacionando Ventas..." />;

  if (data.error) {
    return (
      <div className="p-8 h-full">
        <NoDataMessage message={data.error} isError={true} />
      </div>
    );
  }

  // Usamos los datos de salesAdoption que se generan en DataService
  const salesData = data.salesAdoption.total; 

  // ORDEN ECONÓMICO (de mayor a menor dinero)
const SALES_ORDER = [
    "Grande - > $50.000 M",
    "Mediana (alta) - $10.000-50.000 M",
    "Mediana (baja) - $1.000-10.000 M",
    "Pequeña - < $1.000 M"
];

// Reordenar según SALES_ORDER
salesData.sort((a, b) => {
  return SALES_ORDER.indexOf(a.name) - SALES_ORDER.indexOf(b.name);
});


  if (data.empty || salesData.length === 0) {
    return <div className="p-8 h-full"><NoDataMessage message={data.empty || "No hay datos de adopción para rangos de ventas."} isError={false} /></div>;
  }

  return (
    <div className="h-full flex flex-col p-8 animate-fadeIn">
      <SectionTitle title="Adopción por Volumen de Ventas" subtitle="Impacto del tamaño de facturación en la madurez tecnológica" />
      <Card className="flex-1 p-8">
        <ResponsiveContainer width="100%" height="100%">
          {/* Gráfico de Barras Horizontal Apilado (100% Stacked Bar Chart) */}
          <BarChart layout="vertical" data={salesData} margin={{ top: 20, right: 45, left: 25, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            {/* Eje X (Horizontal) para los valores porcentuales */}
            <XAxis
              type="number"
              domain={[0, 100]}                // ⬅ forzamos 0–100
              ticks={[0, 25, 50, 75, 100]}     // ⬅ mismas marcas pero sin 120
              tickFormatter={(v) => `${v}%`}   // ⬅ añadimos el símbolo %
            />
 
            {/* Eje Y (Vertical) para los rangos de ventas */}
            <YAxis 
              dataKey="name" 
              type="category" 
              width={300} 
              tick={{fill: '#666', fontWeight: 600}} 
            /> 
            <RechartsTooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
              cursor={{ fill: 'rgba(0,0,0,0.05)' }}
            />
            <Legend
              verticalAlign="top"
              height={70}
              content={() => (
                <div
                  style={{
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    gap: "35px",
                    width: "100%",
                    marginBottom: "15px",
                  }}
                >
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Bajo'] }}></span>
                    Bajo
                  </span>

                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Medio'] }}></span>
                    Medio
                  </span>

                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Alto'] }}></span>
                    Alto
                  </span>

                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ width: 12, height: 12, background: PALETTE.levels['Avanzado'] }}></span>
                    Avanzado
                  </span>
                </div>
              )}
            />
            {['Bajo', 'Medio', 'Alto', 'Avanzado'].map((key) => (
              <Bar 
                key={key} 
                dataKey={key} 
                stackId="a" 
                fill={PALETTE.levels[key]} 
                barSize={40}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
};

// ---- SLIDE 4: TECNOLOGÍAS UTILIZADAS ------------------------------

const Block4 = ({ isActive }) => {
  const [techs, setTechs] = useState(null);
  const [error, setError] = useState(null);
  const [emptyMsg, setEmptyMsg] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Solo cargamos una vez cuando el slide se active
    if (!isActive || techs !== null || loading) return;

    const fetchTechs = async () => {
      setLoading(true);
      setError(null);
      setEmptyMsg(null);

      if (!SUPABASE_URL || !SUPABASE_KEY) {
        console.error("[Block4] Faltan VITE_SUPABASE_URL o VITE_SUPABASE_KEY");
        setError(
          "Faltan las credenciales de Supabase (VITE_SUPABASE_URL / VITE_SUPABASE_KEY)."
        );
        setTechs([]);
        setLoading(false);
        return;
      }

      try {
        const ENDPOINT_TECHS = `${SUPABASE_URL}/rest/v1/techs`;

        const params = new URLSearchParams({
          select: "tecnologia,adopcion", // 👈 columnas reales
        });
        params.append("order", "adopcion.desc");

        const url = `${ENDPOINT_TECHS}?${params.toString()}`;
        const headers = {
          apikey: SUPABASE_KEY,
          Authorization: `Bearer ${SUPABASE_KEY}`,
          Accept: "application/json",
        };

        console.log("▶️ [Block4] Llamando a:", url);

        const response = await fetch(url, { headers });
        const rawText = await response.text();
        console.log(
          "📡 [Block4] Status:",
          response.status,
          response.statusText,
          "Body:",
          rawText
        );

        if (!response.ok) {
          setError(
            `Error HTTP ${response.status} - ${response.statusText}. Respuesta del servidor: ${rawText || "(sin cuerpo)"}`
          );
          setTechs([]);
          setLoading(false);
          return;
        }

        const data = rawText ? JSON.parse(rawText) : [];

        // 🔍 Mapeo sencillo y robusto
        const mapped = (data || []).map((row) => ({
          name: row.tecnologia || "",                         // 👈 OBLIGAMOS a usar 'tecnologia'
          value: parseFloat(row.adopcion ?? 0) || 0,          // en porcentaje
        }));

        console.log("[Block4] Mapped data:", mapped);

        // si realmente viene vacío
        if (!mapped.length || !mapped.some((t) => t.name)) {
          setEmptyMsg("No se encontraron registros en la tabla 'techs'.");
        }

        setTechs(mapped);
      } catch (err) {
        console.error("❌ [Block4] Error cargando tecnologías:", err);
        setError(
          `No se pudieron cargar las tecnologías desde la base de datos. Detalle: ${
            err?.message || String(err)
          }`
        );
        setTechs([]);
      } finally {
        setLoading(false);
      }
    };

    fetchTechs();
  }, [isActive, techs, loading]);

  // --- RENDER ---

  if (loading || techs === null) {
    return <LoadingOverlay text="Cargando Tecnologías Utilizadas..." />;
  }

  if (error) {
    return (
      <div className="p-8 h-full">
        <NoDataMessage message={error} isError={true} />
      </div>
    );
  }

  // Solo mostramos "sin datos" si de verdad no hay ninguna tecnología con nombre
  const visibleTechs = (techs || []).filter((t) => t.name);

  if (emptyMsg || !visibleTechs.length) {
    return (
      <div className="p-8 h-full">
        <NoDataMessage
          message={emptyMsg || "No se encontraron registros en la tabla 'techs'."}
          isError={false}
        />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-8 animate-fadeIn">
      <SectionTitle
        title="Tecnologías Utilizadas"
        subtitle="Porcentaje de empresas que declara usar cada tecnología"
      />

      <Card className="flex-1 p-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={visibleTechs}
            layout="vertical"
            margin={{ top: 20, right: 50, left: 40, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            {/* eje Y: nombre de la tecnología */}
            <YAxis
              dataKey="name"
              type="category"
              tick={{ fill: "#555", fontSize: 12 }}
              width={180}
            />
            {/* eje X: porcentaje */}
            <XAxis
              type="number"
              domain={[0, "dataMax + 5"]}
              tickFormatter={(v) => `${v}%`}
            />
            <RechartsTooltip
              formatter={(value) => [`${value}%`, "Adopción"]}
              labelFormatter={(label) => label}
              contentStyle={{
                borderRadius: "8px",
                border: "none",
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              }}
            />
            <Bar
              dataKey="value"
              fill={PALETTE.primary}
              barSize={24}
              radius={[4, 4, 4, 4]}
              // etiqueta al final de cada barra, como en el mock
              label={({ x, y, width, value }) => (
                <text
                  x={x + width + 8}
                  y={y + 14}
                  fill="#444"
                  fontSize={12}
                >
                  {`${value}%`}
                </text>
              )}
            />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
};


// --- SLIDE 5 (NUBE DE PALABRAS)
// --- SLIDE 5 (NUBE DE PALABRAS) ---
// --- BLOQUE 5: Nube de palabras (tabla `nube_palabras`) ---
const Block5 = ({ isActive }) => {
  const [words, setWords] = useState(null);        // datos de la nube
  const [error, setError] = useState(null);        // mensaje de error (si lo hay)
  const [emptyMsg, setEmptyMsg] = useState(null);  // mensaje de "sin datos"
  const [loading, setLoading] = useState(false);   // estado de carga

  useEffect(() => {
    if (!isActive || words !== null || loading) return;

    const fetchWordCloud = async () => {
      setLoading(true);
      setError(null);
      setEmptyMsg(null);

      if (!SUPABASE_URL || !SUPABASE_KEY) {
        console.error("[Block5] Faltan VITE_SUPABASE_URL o VITE_SUPABASE_KEY");
        setError(
          "Faltan las credenciales de Supabase (VITE_SUPABASE_URL / VITE_SUPABASE_KEY)."
        );
        setWords([]);
        setLoading(false);
        return;
      }

      try {
        const WORDS_ENDPOINT = `${SUPABASE_URL}/rest/v1/nube_palabras`;

        const params = new URLSearchParams({
          select: "palabra,frecuencia",
        });
        params.append("order", "frecuencia.desc");

        const url = `${WORDS_ENDPOINT}?${params.toString()}`;
        const headers = {
          apikey: SUPABASE_KEY,
          Authorization: `Bearer ${SUPABASE_KEY}`,
          Accept: "application/json",
        };

        console.log("▶️ [Block5] Llamando a:", url);

        const response = await fetch(url, { headers });
        const rawText = await response.text();
        console.log(
          "📡 [Block5] Status:",
          response.status,
          response.statusText,
          "Body:",
          rawText
        );

        if (!response.ok) {
          setError(
            `Error HTTP ${response.status} - ${response.statusText}. Respuesta del servidor: ${rawText || "(sin cuerpo)"}`
          );
          setWords([]);
          setLoading(false);
          return;
        }

        const data = rawText ? JSON.parse(rawText) : [];

        const mapped = (data || [])
          .map((row) => ({
            text: (row.palabra || "").toUpperCase(),
            value: Number(row.frecuencia) || 0,
          }))
          .filter((w) => w.text && w.value > 0);

        if (!mapped.length) {
          setEmptyMsg("No se encontraron registros en la tabla 'nube_palabras'.");
        }

        setWords(mapped);
      } catch (err) {
        console.error("❌ [Block5] Error cargando nube de palabras:", err);
        setError(
          `No se pudo cargar la nube de palabras desde la base de datos. Detalle: ${
            err?.message || String(err)
          }`
        );
        setWords([]);
      } finally {
        setLoading(false);
      }
    };

    fetchWordCloud();
  }, [isActive, words, loading]);

  // --- RENDER ---

  if (loading || words === null) {
    return (
      <LoadingOverlay text="Procesando Nube de Palabras... Intentando conexión con la API REST de Supabase..." />
    );
  }

  if (error) {
    return (
      <div className="p-8 h-full">
        <NoDataMessage message={error} isError={true} tableName="nube_palabras" />
      </div>
    );
  }

  if (emptyMsg || !words.length) {
    return (
      <div className="p-8 h-full">
        <NoDataMessage
          message={emptyMsg || "No se encontraron registros en la tabla 'nube_palabras'."}
          isError={false}
          tableName="nube_palabras"
        />
      </div>
    );
  }

  // Calcular min y max para escalar tamaños
  const maxValue = Math.max(...words.map((w) => w.value));
  const minValue = Math.min(...words.map((w) => w.value));
  const minFont = 16; // px
  const maxFont = 60; // px (más grande que antes para que destaque más)

  return (
    <div className="h-full flex flex-col p-8 animate-fadeIn">
      <SectionTitle
        title="Propósito Empresarial Actual"
        subtitle="Palabras más usadas en la encuesta"
      />

      {/* overflow-auto para que nunca corte palabras, sin el texto de pie de página */}
      <Card className="flex-1 relative flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 overflow-auto">
        <div className="flex flex-wrap justify-center items-center content-center gap-4 max-w-5xl p-10">
          {words.map((word, i) => {
            // normalizamos 0..1 según frecuencia
            const range = Math.max(1, maxValue - minValue);
            const norm = (word.value - minValue) / range;

            // tamaño entre minFont y maxFont, con más contraste
            const fontSize = minFont + norm * (maxFont - minFont);
            const opacity = 0.3 + norm * 0.7; // entre 0.3 y 1

            const color =
              i % 3 === 0
                ? PALETTE.primary
                : i % 3 === 1
                ? PALETTE.secondary
                : "#555";

            return (
              <span
                key={`${word.text}-${i}`}
                className="cursor-default hover:scale-110 transition-transform duration-300 font-bold inline-block"
                style={{
                  fontSize: `${fontSize}px`,
                  color,
                  opacity,
                }}
                title={`${word.text}: ${word.value} menciones`}
              >
                {word.text}
              </span>
            );
          })}
        </div>
      </Card>
    </div>
  );
};

 

 
// --- BLOQUE 6 (NUEVA LÍNEA DE TIEMPO CON EFECTO WOW) ---
const TimelineItem = ({ data, index }) => {
    const Icon = data.icon;
    
    // Lógica Invertida: HOY (0) y +18 AÑOS (2) a la izquierda. EVOLUCIÓN (1) a la derecha.
    // isLeft = TRUE para índices pares (0, 2...)
    const isLeft = index % 2 === 0; 
 
    return (
        <div 
            // Margen Superior Negativo para superposición
            className={`flex w-full animate-slideUp items-start h-[33svh] ${index > 0 ? 'mt-[-60px] md:mt-[-76px]' : ''}`} // Ajuste de margen para superponer items
            style={{ animationDelay: `${index * 300 + 500}ms` }}
        >
            
            {/* Columna de Contenido (Texto de IA) */}
            <div className={`w-1/2 ${isLeft ? 'pr-4 sm:pr-8' : 'pl-4 sm:pl-8'} ${isLeft ? 'order-1' : 'order-3'}`}> 
                <div className={`bg-white p-4 rounded-xl shadow-2xl hover:shadow-3xl transition-shadow duration-300 h-full flex flex-col justify-between overflow-hidden ${isLeft ? 'text-right border-r-4' : 'text-left border-l-4'} border-[#E30613]`}>
                    
                    {/* Título del Período (Alineado con el contenido) */}
                    <h3 className={`text-xl md:text-xl font-bold text-[#E30613] mb-2 ${isLeft ? 'self-end' : 'self-start'}`}>
                        {data.period}
                    </h3>
 
                    {/* Descripción de IA */}
                    <p className="text-sm md:text-base text-gray-700 leading-tight">
                      {data.description}
                    </p>

                    
                    {/* Énfasis en la generación por IA */}
                    <span className={`inline-flex items-center mt-2 text-xs font-semibold text-[#BA0C2F] bg-red-50 px-2 py-0.5 rounded-full ${isLeft ? 'self-end' : 'self-start'}`}>
                        <Brain size={10} className="mr-1" />
                        Síntesis de IA
                    </span>
                </div>
            </div>
 
            {/* Línea de Tiempo y Nodo Central (Fijo en el centro) */}
            <div className="w-8 flex flex-col items-center order-2">
                {/* Círculo del Nodo */}
                <div className="relative z-10 w-6 h-6 rounded-full bg-[#E30613] flex items-center justify-center shadow-xl transform hover:scale-110 transition-transform duration-300">
                    <Icon size={14} className="text-white" />
                </div>
            </div>

            {/* Columna de Espaciado (Vacío) - solo se usa para forzar el layout */}
            <div className={`w-1/2 ${isLeft ? 'order-3' : 'order-1'}`}>
                {/* Este div está vacío, pero su orden asegura que la caja de contenido ocupe el lado correcto */}
            </div>
            
        </div>
    );
};

const Block6 = ({ isActive }) => {
  const [loading, setLoading] = useState(true);
  const [llmData, setLlmData] = useState(null);

  // ---- FETCH A SUPABASE ----
  const fetchLlmData = async () => {
    if (!SUPABASE_URL || !SUPABASE_KEY) {
      console.error("Faltan credenciales de Supabase.");
      setLlmData(null);
      setLoading(false);
      return;
    }

    const ENDPOINT_LLM = `${BASE_URL}/rest/v1/llm?select=pregunta_1,pregunta_2,resumen_final&order=created_at.desc&limit=1`;

    try {
      const response = await fetch(ENDPOINT_LLM, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          apikey: SUPABASE_KEY,
          Authorization: `Bearer ${SUPABASE_KEY}`,
        },
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      setLlmData(data?.[0] || {});
    } catch (err) {
      console.error("Error al leer Supabase:", err);
      setLlmData({});
    } finally {
      setLoading(false);
    }
  };

  // ---- TYPEWRITER ----
  const typeWriter = (element, text, speed = 15) => {
    return new Promise((resolve) => {
      let i = 0;
      element.innerHTML = "";

      function typing() {
        if (i < text.length) {
          element.innerHTML += text.charAt(i);
          i++;
          setTimeout(typing, speed);
        } else {
          resolve();
        }
      }
      typing();
    });
  };

  // ---- SECUENCIA DE ESCRITURA ----
  const startSequence = async () => {
    const elA = document.getElementById("visionActual");
    const elB = document.getElementById("vision18");
    const elC = document.getElementById("evolucion");

    if (!elA || !elB || !elC) return;

    await typeWriter(elA, llmData.pregunta_1 || "");
    await typeWriter(elB, llmData.pregunta_2 || "");
    await typeWriter(elC, llmData.resumen_final || "");
  };

  // ---- CARGA CUANDO SE ACTIVA EL SLIDE ----
  useEffect(() => {
    if (isActive) {
      setLoading(true);
      const timer = setTimeout(async () => {
        await fetchLlmData();
      }, 1500);

      return () => clearTimeout(timer);
    }
  }, [isActive]);

  // ---- INICIAR TYPEWRITER CUANDO YA HAY DATOS ----
  useEffect(() => {
    if (!loading && llmData) startSequence();
  }, [loading, llmData]);

  // ---- LOADING ----
  if (loading)
    return (
      <div className="h-full flex items-center justify-center bg-gray-900">
        <LoadingOverlayBlock6 />
      </div>
    );

  // ---- NO DATA ----
  if (!llmData || Object.keys(llmData).length === 0)
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <NoDataMessage />
      </div>
    );

  // ---- UI NUEVA (70% / 30%) ----
  return (
    <div className="h-full w-full grid grid-cols-1 lg:grid-cols-10 gap-4 p-6 bg-gray-50">

      {/* Columna izquierda (70%) */}
      <div className="col-span-1 lg:col-span-7 flex flex-col gap-4">

        {/* Visión Actual */}
        <div className="flex-1 bg-white p-6 rounded-xl shadow-xl h-full">
          <h2 className="text-xl font-bold text-gray-800 border-b pb-2 mb-4">
            Propósito Actual
          </h2>
          <p id="visionActual" className="text-gray-700 text-lg leading-relaxed"></p>
        </div>

        {/* Visión 18 Años */}
        <div className="flex-1 bg-white p-6 rounded-xl shadow-xl h-full">
          <h2 className="text-xl font-bold text-gray-800 border-b pb-2 mb-4">
            Propósito en 18 Años
          </h2>
          <p id="vision18" className="text-gray-700 text-lg leading-relaxed"></p>
        </div>
      </div>

      {/* Columna derecha (30%) */}
      <div className="col-span-1 lg:col-span-3 bg-red-700 p-8 rounded-xl shadow-2xl flex flex-col h-full">
        <h2 className="text-2xl font-bold text-white border-b border-red-500 pb-3 mb-6">
          Evolución de propósito
        </h2>
        <p id="evolucion" className="text-white text-lg leading-relaxed"></p>
      </div>
    </div>
  );
};



// ======================================================
// --- SUBCOMPONENTE PARA PANEL DE VISIÓN ---
// ======================================================
const VisionPanel = ({ title, color, content, panelId }) => {
  const textColor =
    color === "red" ? "text-white" : "text-gray-800";
  const bg =
    color === "red" ? "bg-red-700 hover:shadow-red-900" : "bg-white hover:shadow-blue-300";
  const borderIcon =
    color === "red" ? "text-red-300" : "text-blue-500";

  return (
    <div className={`${bg} p-6 rounded-xl shadow-2xl transition-all flex flex-col`}>
      <h2 className={`text-2xl font-bold border-b-2 pb-3 mb-4 flex items-center ${textColor}`}>
        <svg className={`w-6 h-6 mr-2 ${borderIcon}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3" />
        </svg>
        {title}
      </h2>

      <p className={`${textColor} text-lg leading-relaxed whitespace-pre-line`}>
        {content}
      </p>
    </div>
  );
};


export default function DashboardApp() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const totalSlides = 7; // Reducido a 3: Intro, Block1, Block2, Block6
 
  const nextSlide = () => setCurrentSlide(p => Math.min(p + 1, totalSlides - 1));
  const prevSlide = () => setCurrentSlide(p => Math.max(p - 1, 0));
 
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowRight') nextSlide();
      if (e.key === 'ArrowLeft') prevSlide();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, []);
 
  return (
    <div className="w-full h-screen bg-gray-100 flex flex-col font-sans overflow-hidden">
      <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8 shadow-sm z-50">
  {/* LOGO CÁMARA DE COMERCIO DE BOGOTÁ */}
  <img
    src="/Cámara_de_Comercio_de_Bogotá_logo.png"
    alt="Cámara de Comercio de Bogotá"
    className="h-10 object-contain"
  />

  {/* LOGO STRATESYS EN LA DERECHA */}
  <img
    src="/Stratesys.png"
    alt="Stratesys"
    className="h-8 opacity-90 hover:opacity-100 transition-opacity"
  />
</header>

 
      <main className="flex-1 relative overflow-hidden">
        {currentSlide === 0 && <IntroSlide onNext={nextSlide} />}
        {currentSlide === 1 && <Block1 isActive={true} />}
        {currentSlide === 2 && <Block2 isActive={true} />}
        {currentSlide === 3 && <Block3 isActive={true} />}
        {currentSlide === 4 && <Block4 isActive={true} />}
        {currentSlide === 5 && <Block5 isActive={true} />}
        {currentSlide === 6 && <Block6 isActive={true} />}
      </main>
 
      <div className="absolute bottom-8 right-8 flex gap-4 z-50">
        <button
          onClick={prevSlide}
          disabled={currentSlide === 0}
          className="p-3 rounded-full bg-white shadow-lg text-gray-600 hover:text-[#E30613] disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          <ChevronLeft />
        </button>
        <button
          onClick={nextSlide}
          disabled={currentSlide === totalSlides - 1}
          className="p-3 rounded-full bg-[#E30613] shadow-lg text-white hover:bg-[#BA0C2F] disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          <ChevronRight />
        </button>
      </div>
 
      {/* FIX: Se eliminan los atributos 'jsx' y 'global' del tag <style> para evitar la advertencia de React sobre atributos no booleanos. */}
      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { transform: translateY(0); } }
        @keyframes progress { 0% { width: 0%; } 100% { width: 100%; } }
        .animate-fadeIn { animation: fadeIn 0.8s ease-out forwards; }
        .animate-slideUp { animation: slideUp 0.8s ease-out forwards; }
        .animate-progress { animation: progress 2s ease-in-out forwards; }
        .text-shadow-sm { text-shadow: 1px 1px 2px rgba(0,0,0,0.3); }
      `}</style>
    </div>
  );
}
 
