import React, { useState } from 'react';
 
// Se asume que Tailwind CSS está disponible en el entorno de ejecución,
// por lo que usamos sus clases directamente.
 
// Variables de color (Clases de Tailwind personalizadas para el branding)
// Rojo Principal: #E30613 -> ccb-red
// Rojo Secundario: #BA0C2F -> ccb-red-dark
// Gris Fondo: #F9F9F9 -> gray-bg
 
// Definición de Íconos SVG (usando Lucide Icons o SVGs simples)
 
const QrIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7"></rect>
    <rect x="14" y="3" width="7" height="7"></rect>
    <rect x="14" y="14" width="7" height="7"></rect>
    <rect x="3" y="14" width="7" height="7"></rect>
  </svg>
);
 
const SurveyIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>
    <path d="M22 12A10 10 0 0 0 12 2v10z"></path>
  </svg>
);
 
const PlayIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="5 3 19 12 5 21 5 3"></polygon>
  </svg>
);
 
const ResultsIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"></line>
    <line x1="12" y1="20" x2="12" y2="4"></line>
    <line x1="6" y1="20" x2="6" y2="14"></line>
  </svg>
);
 
const ArrowRightIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="9 18 15 12 9 6"></polyline>
  </svg>
);
 
const ArrowLeftIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="15 18 9 12 15 6"></polyline>
  </svg>
);
 
const BackIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="12" x2="5" y2="12"></line>
    <polyline points="12 19 5 12 12 5"></polyline>
  </svg>
);
 
/**
 * Componente funcional para un botón de comando individual.
 * @param {object} props - Propiedades del componente.
 * @param {string} props.id - ID único para el botón.
 * @param {string} props.title - Título principal del botón.
 * @param {string} props.description - Descripción del comando.
 * @param {string} props.command - Comando a enviar.
 * @param {function} props.onSend - Función de callback para enviar el comando.
 * @param {React.Component} props.Icon - Componente de ícono SVG.
 * @param {boolean} props.hasArrow - Indica si debe mostrar la flecha de navegación.
 */
const CommandButton = ({ id, title, description, command, onSend, Icon, hasArrow = false }) => {
  // Manejador de click que envía el comando y, si aplica, cambia la vista.
  const handleClick = () => {
    onSend(command);
  };
 
  return (
    <button
      id={id}
      onClick={handleClick}
      className="bg-white border-l-4 border-[#E30613] rounded-xl p-4 flex items-center justify-between shadow-lg hover:shadow-xl active:scale-[0.98] active:bg-[#fff5f5] transition duration-150 ease-in-out cursor-pointer overflow-hidden"
    >
      <div className="flex items-center gap-4 text-left">
        <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center text-[#E30613] flex-shrink-0">
          <Icon className="w-5 h-5" />
        </div>
        <div className="flex flex-col">
          <h3 className="text-base font-semibold text-[#333333] mb-0">{title}</h3>
          <p className="text-xs text-[#666666]">{description}</p>
        </div>
      </div>
      {hasArrow && (
        <ArrowRightIcon className="w-5 h-5 text-gray-300 ml-4 flex-shrink-0" strokeWidth="2" />
      )}
    </button>
  );
};
 
// --- VISTAS PRINCIPALES ---
 
/**
 * Vista 1: Menú Principal con los 4 botones.
 */
const MainView = ({ onCommand, onSwitchToNav }) => (
  <div className="grid grid-cols-1 gap-4 w-full animate-fadeIn">
    <CommandButton
      id="btn-show-qr"
      title="Mostrar QR"
      description="Proyectar código de acceso"
      command="SHOW_QR"
      onSend={onCommand}
      Icon={QrIcon}
    />
    <CommandButton
      id="btn-survey-dashboard"
      title="Encuesta + Live"
      description="QR Encuesta y estadísticas"
      command="SHOW_SURVEY"
      onSend={onCommand}
      Icon={SurveyIcon}
    />
    <CommandButton
      id="btn-run-analysis"
      title="Ejecutar Análisis"
      description="Procesar datos en remoto"
      command="RUN_ANALYSIS"
      onSend={onCommand}
      Icon={PlayIcon}
    />
    <CommandButton
      id="btn-view-results"
      title="Ver Resultados"
      description="Abrir Dashboard final"
      command="OPEN_RESULTS_URL"
      onSend={onSwitchToNav} // Al pulsar, envía el comando Y cambia de vista
      Icon={ResultsIcon}
      hasArrow={true}
    />
  </div>
);
 
/**
 * Vista 2: Controles de Navegación (Anterior/Siguiente).
 */
const NavView = ({ onCommand, onReturnToMain }) => (
  <div className="flex flex-col gap-6 items-center justify-center w-full min-h-[calc(100vh-70px-40px-60px)] animate-fadeIn">
   
    <div className="text-center mb-4">
      <h2 className="text-xl font-bold text-[#E30613]">Control de Resultados</h2>
      <p className="text-sm text-[#666666]">Navegar por las diapositivas</p>
    </div>
 
    <div className="flex gap-4 w-full max-w-sm">
      {/* Botón Anterior */}
      <button
        id="btn-prev"
        onClick={() => onCommand('NAV_PREV')}
        className="flex-1 bg-white text-[#E30613] border-2 border-[#E30613] p-6 rounded-xl text-3xl flex justify-center items-center shadow-md hover:bg-[#E30613] hover:text-white transition duration-150 ease-in-out h-24"
      >
        <ArrowLeftIcon className="w-8 h-8" strokeWidth="3" />
      </button>
 
      {/* Botón Siguiente */}
      <button
        id="btn-next"
        onClick={() => onCommand('NAV_NEXT')}
        className="flex-1 bg-white text-[#E30613] border-2 border-[#E30613] p-6 rounded-xl text-3xl flex justify-center items-center shadow-md hover:bg-[#E30613] hover:text-white transition duration-150 ease-in-out h-24"
      >
        <ArrowRightIcon className="w-8 h-8" strokeWidth="3" />
      </button>
    </div>
 
    {/* Botón Volver */}
    <button
      id="btn-back"
      onClick={onReturnToMain}
      className="mt-4 flex items-center gap-2 text-[#666666] font-semibold text-sm hover:text-[#BA0C2F] transition duration-150"
    >
      <BackIcon className="w-5 h-5" />
      Volver al menú
    </button>
  </div>
);
 
 
/**
 * Componente Principal de la Aplicación (App)
 */
const App = () => {
  // Estado para controlar qué vista se muestra: 'main' o 'nav'
  const [currentView, setCurrentView] = useState('main');
 
  /**
   * Función central para enviar comandos. Simula una llamada API/WebSocket.
   * @param {string} command - El ID del comando a ejecutar (e.g., 'SHOW_QR').
   */
  const sendCommand = (command) => {
    console.log(`[COMANDO ENVIADO]: ${command}`);
   
    // Feedback visual (Vibración si el dispositivo lo permite)
    if (navigator.vibrate) {
        navigator.vibrate(50);
    }
 
    // --- INTEGRACIÓN: Aquí iría la lógica de red (fetch/socket) ---
    // Example: fetch('/api/control', { method: 'POST', body: JSON.stringify({ action: command }) });
  };
 
  /**
   * Maneja el cambio a la vista de navegación.
   * Llama a sendCommand para abrir la URL antes de cambiar la UI.
   */
  const handleSwitchToNav = () => {
    sendCommand('OPEN_RESULTS_URL');
    setCurrentView('nav');
  };
 
  /**
   * Maneja el regreso a la vista principal.
   */
  const handleReturnToMain = () => {
    // sendCommand('CLOSE_RESULTS'); // Opcional
    setCurrentView('main');
  };
 
  return (
    <div className="min-h-screen bg-[#F9F9F9] flex flex-col font-poppins">
     
      {/* HEADER */}
      <header className="sticky top-0 h-[70px] bg-white flex justify-between items-center px-5 shadow-md z-10">
        <div className="flex items-center gap-2 font-bold text-sm text-[#E30613]">
          <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: '#E30613' }}>
            <rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect>
            <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path>
          </svg>
          CCB
        </div>
        <div className="font-semibold text-xs text-gray-600 tracking-wide">
          STRATESYS
        </div>
      </header>
 
      {/* MAIN CONTENT */}
      <main className="flex-grow flex flex-col p-5 pt-10 max-w-xl w-full mx-auto relative">
        <div className="text-center mb-6 text-sm text-[#666666] uppercase tracking-wider">
          III Encuentro Trayectoria MEGA
        </div>
 
        {/* Renderizado condicional de las vistas */}
        {currentView === 'main' && (
          <MainView
            onCommand={sendCommand}
            onSwitchToNav={handleSwitchToNav}
          />
        )}
       
        {currentView === 'nav' && (
          <NavView
            onCommand={sendCommand}
            onReturnToMain={handleReturnToMain}
          />
        )}
      </main>
 
      {/* FOOTER */}
      <footer className="text-center p-4 text-xs text-gray-400 bg-[#F9F9F9] w-full mt-auto">
        &copy; 2025 CCB & Stratesys - Event Technology
      </footer>
    </div>
  );
};
 
export default App;
 