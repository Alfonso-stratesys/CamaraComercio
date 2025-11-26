📊 Dashboard Proyecto MEGA - Visión Empresarial 2025

Este proyecto es un dashboard interactivo desarrollado para visualizar los resultados de la Encuesta de Visión Empresarial (Cámara de Comercio de Bogotá & Stratesys). Utiliza gráficos dinámicos y animaciones para presentar datos sobre madurez digital, distribución industrial y tecnologías habilitadoras.

🚀 Tecnologías Utilizadas

React (v18+)

Vite (Build tool & Dev Server)

Recharts (Librería de gráficos)

Lucide React (Paquete de iconos)

Tailwind CSS (Estilos y diseño responsivo)

🛠️ Requisitos Previos

Antes de comenzar, asegúrate de tener instalado en tu sistema:

Node.js (Versión 16 o superior recomendada).

Git.

📥 Instalación y Ejecución (Quick Start)

Sigue estos pasos si acabas de clonar este repositorio:

1. Clonar el repositorio

Abre tu terminal y ejecuta:

git clone <URL_DE_TU_REPOSITORIO>
cd front-mega


2. Instalar dependencias

Este paso es crucial. Descargará todas las librerías necesarias (React, Recharts, Tailwind, etc.) listadas en el package.json.

npm install


3. Ejecutar el servidor de desarrollo

Para ver el proyecto en tu navegador:

npm run dev


Haz clic en el enlace que aparece en la terminal (usualmente http://localhost:5173/).

🆘 Solución de Problemas Comunes

Error: "Failed to resolve import 'recharts' or 'lucide-react'"

Si al ejecutar npm run dev ves un error indicando que faltan módulos, significa que las dependencias no se instalaron correctamente. Ejecuta manualmente:

npm install recharts lucide-react


Los estilos se ven "rotos" o feos (Tailwind CSS)

Si la aplicación carga pero no tiene estilos (se ve todo blanco y desordenado), asegúrate de que Tailwind esté configurado.

Instalar Tailwind (si no está):

npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p


Verificar tailwind.config.js:
Asegúrate de que el archivo tenga esta configuración en content:

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}


Verificar src/index.css:
Debe incluir estas tres líneas al principio:

@tailwind base;
@tailwind components;
@tailwind utilities;


📂 Estructura del Proyecto

/
├── public/              # Archivos estáticos
├── src/
│   ├── App.jsx          # Componente principal (Dashboard)
│   ├── index.css        # Estilos globales y directivas de Tailwind
│   └── main.jsx         # Punto de entrada de React
├── package.json         # Lista de dependencias y scripts
├── tailwind.config.js   # Configuración de estilos
└── vite.config.js       # Configuración del compilador


📜 Scripts Disponibles

npm run dev: Inicia el servidor de desarrollo.

npm run build: Compila la aplicación para producción.

npm run preview: Vista previa local de la build de producción.