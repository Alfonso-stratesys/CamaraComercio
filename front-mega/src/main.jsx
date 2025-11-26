// import { StrictMode } from 'react'
// import { createRoot } from 'react-dom/client'
// import './index.css'
// import Graficos from './Graficos.jsx'

// createRoot(document.getElementById('root')).render(
//   <StrictMode>
//     <Graficos />
//   </StrictMode>,
// )
import React, { StrictMode } from 'react';
import { createRoot } from "react-dom/client";
import "./index.css";
// IMPORTANTE: Solo importamos BrowserRouter para envolver la app.
import { BrowserRouter } from 'react-router-dom'; 
// Importa el componente que maneja las rutas
import Rutas from "./Rutas.jsx"; 

createRoot(document.getElementById("root")).render(
  // La estructura de renderizado estándar de React
  <StrictMode>
    <BrowserRouter>
      {/* Llama a tu componente de Rutas */}
      <Rutas /> 
    </BrowserRouter>
  </StrictMode>
);

// NOTA: Elimina cualquier otra importación o código que intente renderizar 
// Graficos o Formulario directamente en este archivo.