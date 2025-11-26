import React from 'react';
// IMPORTANTE: Estos componentes de React Router deben venir de 'react-router-dom'
import { Routes, Route } from 'react-router-dom';

// Importa tus componentes
import Formulario from './Formulario.jsx'; 
import Graficos from './Graficos.jsx'; 
import DashBoard from './Dashboard_vivo.jsx';
import Admin from './Panel_admin.jsx';  

// Este componente define qué se muestra en cada URL
export default function Rutas() {
  return (
    <Routes>
      {/* 1. Ruta para el dashboard (http://localhost:XXXX/ o /Graficos) */}
      <Route path="/" element={<Graficos />} />
      <Route path="/Graficos" element={<Graficos />} /> 

      {/* 2. Ruta para el formulario (http://localhost:XXXX/Formulario) */}
      <Route path="/Formulario" element={<Formulario />} />
      <Route path="/Dashboard" element={<DashBoard />} />
      <Route path="/Admin" element={<Admin />} />
      {/* Opcional: Ruta para manejar 404s */}
      <Route path="*" element={
        <div style={{ padding: '50px', textAlign: 'center', minHeight: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
          <h1>404 | Página No Encontrada</h1>
          <p>La ruta que intentas cargar no existe. Verifica la URL.</p>
        </div>
      } />
    </Routes>
  );
}