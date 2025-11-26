import React, { useState, useEffect } from 'react';
import { 
    Building2, 
    Users, 
    DollarSign, 
    TrendingUp, 
    Cpu, 
    Lightbulb, 
    CheckCircle2, 
    ArrowRight, 
    ArrowLeft, 
    BarChart3, 
    Target,
    Check
} from 'lucide-react';

// 1. CONFIGURACIÓN DE SUPABASE
const supabaseUrl = 'https://xfivwryjelypnbmwoaan.supabase.co';
// IMPORTANTE: Esta es la clave que debe ser validada por el usuario. 
// La he restaurado al valor original que me diste. Si sigue fallando, 
// el usuario debe reemplazarla con la clave 'anon public' de su dashboard.
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhmaXZ3cnlqZWx5cG5ibXdvYWFuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM2MzczOTMsImV4cCI6MjA3OTIxMzM5M30.D8f1IFZ5sVLLT4M6wfy5yjUOKT7H60Tv2UXV2i5gPYU';
// const supabaseUrl = process.env.REACT_APP_SUPABASE_URL ?? null;
// const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY ?? null;
// const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || "";
// const supabaseAnonKey = import.meta.env.VITE_SUPABASE_KEY || "";

export default function App() {
    const [supabaseClient, setSupabaseClient] = useState(null);
    const [isClientLoading, setIsClientLoading] = useState(true); 
    
    const [step, setStep] = useState(1);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [submitStatus, setSubmitStatus] = useState(null); // 'success', 'error', null
    const [errors, setErrors] = useState({});
    
    // Estado del formulario
    const [formData, setFormData] = useState({
        industry: '',
        salesVolume: '',
        employees: '',
        techAdoption: '',
        technologies: [],
        purposeCurrent: '',
        purposeFuture: ''
    });

    // Carga de script y inicialización de cliente
    useEffect(() => {
        const scriptUrl = 'https://unpkg.com/@supabase/supabase-js@2.44.2/dist/umd/supabase.js';
        const script = document.createElement('script');
        script.src = scriptUrl;
        
        script.onload = () => {
            try {
                if (window.supabase && window.supabase.createClient) {
                    const client = window.supabase.createClient(supabaseUrl, supabaseAnonKey);
                    setSupabaseClient(client);
                    console.log("Supabase client initialized successfully.");
                } else {
                    console.error("Supabase script loaded, but createClient is not available globally. Check CDN path or version.");
                    setSubmitStatus('error');
                }
            } catch (e) {
                console.error("Error during Supabase client creation:", e);
                setSubmitStatus('error');
            }
            setIsClientLoading(false);
        };
        
        script.onerror = (e) => {
            console.error("FAILED TO LOAD SCRIPT:", scriptUrl, e);
            setSubmitStatus('error');
            setIsClientLoading(false);
        };
        
        document.head.appendChild(script);

        // Cleanup: remueve el script si el componente se desmonta 
        return () => {
            const currentScript = document.querySelector(`script[src="${scriptUrl}"]`);
            if (currentScript) {
                document.head.removeChild(currentScript);
            }
        };
    }, []); 

    // Datos estáticos
    const industries = [
        "Agroindustria", "Manufactura", "Comercio", "Tecnología", 
        "Construcción", "Energía y Minería", "Salud", "Servicios", "Otra"
    ];

    const salesVolumes = [
        "Pequeña - < $1.000 M",
        "Mediana (baja) - $1.000-10.000 M",
        "Mediana (alta) - $10.000-50.000 M",
        "Grande - > $50.000 M"
    ];

    const employeeRanges = [
        "1-50",
        "51-200",
        "201-500 ",
        ">500"
    ];

    const techLevels = [
        { id: 'bajo', label: 'Bajo - Uso limitado de herramientas básicas', color: 'text-red-600', border: 'border-red-200', bg: 'bg-red-50' },
        { id: 'medio', label: 'Medio - Digitalización parcial de procesos', color: 'text-yellow-600', border: 'border-yellow-200', bg: 'bg-yellow-50' },
        { id: 'alto', label: 'Alto - Automatización y analítica integrada', color: 'text-green-600', border: 'border-green-200', bg: 'bg-green-50' },
        { id: 'avanzado', label: 'Avanzado - Uso intensivo de IA, IoT, etc.', color: 'text-blue-600', border: 'border-blue-200', bg: 'bg-blue-50' }
    ];

    const techList = [
        "Inteligencia Artificial", "Automatización / Robótica", "Big Data y analítica avanzada",
        "Internet de las cosas (IoT)", "Computación en la nube", "Ciberseguridad",
        "Blockchain", "Realidad aumentada / virtual", "Impresión 3D", "Ninguna de las anteriores"
    ];

    // Manejadores
    const handleInputChange = (field, value) => {
        setFormData(prev => ({ ...prev, [field]: value }));
        if (errors[field]) {
            setErrors(prev => ({ ...prev, [field]: null }));
        }
    };

    const handleCheckboxChange = (tech) => {
        let updatedTechs = [...formData.technologies];
        if (updatedTechs.includes(tech)) {
            updatedTechs = updatedTechs.filter(t => t !== tech);
        } else {
            if (tech === "Ninguna de las anteriores") {
                updatedTechs = ["Ninguna de las anteriores"];
            } else {
                updatedTechs = updatedTechs.filter(t => t !== "Ninguna de las anteriores");
                updatedTechs.push(tech);
            }
        }
        handleInputChange('technologies', updatedTechs);
    };

    const validateStep = (currentStep) => {
        const newErrors = {};
        let isValid = true;

        if (currentStep === 1) {
            if (!formData.industry) newErrors.industry = "Seleccione una industria";
            if (!formData.salesVolume) newErrors.salesVolume = "Seleccione el volumen de ventas";
            if (!formData.employees) newErrors.employees = "Seleccione el rango de empleados";
        }
        if (currentStep === 2) {
            if (!formData.techAdoption) newErrors.techAdoption = "Seleccione el nivel de adopción";
        }
        if (currentStep === 3) {
            if (!formData.purposeCurrent) newErrors.purposeCurrent = "Este campo es obligatorio";
            if (!formData.purposeFuture) newErrors.purposeFuture = "Este campo es obligatorio";
        }

        if (Object.keys(newErrors).length > 0) {
            setErrors(newErrors);
            isValid = false;
        }
        return isValid;
    };

    const nextStep = () => {
        if (validateStep(step)) {
            setStep(prev => prev + 1);
            window.scrollTo(0, 0);
        }
    };

    const prevStep = () => {
        setStep(prev => prev - 1);
        window.scrollTo(0, 0);
    };

    // FUNCIÓN DE ENVÍO ACTUALIZADA A SUPABASE
    const handleSubmit = async () => {
        if (isClientLoading || !supabaseClient) {
            console.error('Error: Supabase client is not ready. Aborting submission.');
            setSubmitStatus('error');
            setStep(4);
            return;
        }

        if (validateStep(3)) {
            setIsSubmitting(true);
            setSubmitStatus(null);
            
            // ***************************************************************
            // CAMBIO CRUCIAL: Mapeo de campos a columnas de la tabla 'respuestas'
            // ***************************************************************
            const dataToInsert = {
                // frontend_field: backend_column_name
                industria: formData.industry,              // antes era 'industry'
                volumen_ventas: formData.salesVolume,      // antes era 'sales_volume'
                empleados: formData.employees,             // antes era 'employees_range'
                adopcion_tech: formData.techAdoption,      // antes era 'tech_adoption_level'
                techs: formData.technologies,              // antes era 'technologies_used'
                proposito_hoy: formData.purposeCurrent,    // antes era 'purpose_current'
                proposito_18: formData.purposeFuture,      // antes era 'purpose_future'
                created_at: new Date().toISOString()
            };
            
            try {
                // CAMBIO CRUCIAL: Tabla correcta: 'respuestas'
                const { error } = await supabaseClient
                    .from('respuestas') 
                    .insert([dataToInsert]);
                    
                if (error) {
                    console.error('Error al insertar datos en Supabase:', error);
                    // Si el error persiste, la causa es 1) la clave o 2) la RLS
                    if (error.code && error.details) {
                        console.error('Detalles del Error (Verificar RLS):', error.code, error.details);
                    }
                    setSubmitStatus('error');
                } else {
                    setSubmitStatus('success');
                }
            } catch (e) {
                console.error('Error de conexión o inesperado:', e);
                setSubmitStatus('error');
            } finally {
                setIsSubmitting(false);
                setStep(4);
                window.scrollTo(0, 0);
            }
        }
    };

    const resetForm = () => {
        setFormData({
            industry: '',
            salesVolume: '',
            employees: '',
            techAdoption: '',
            technologies: [],
            purposeCurrent: '',
            purposeFuture: ''
        });
        setStep(1);
        setSubmitStatus(null);
    };

    // Header Component
    const Header = () => (
        <header className="fixed top-0 left-0 w-full bg-white shadow-sm z-50 h-20 flex items-center justify-between px-6 md:px-12 border-b border-gray-100">
            <div className="flex items-center gap-4">
                <div className="flex flex-col items-start leading-tight">
                    <span className="font-bold text-gray-800 text-sm tracking-wide">Cámara de Comercio</span>
                    <span className="font-bold text-[#E30613] text-sm tracking-wide">de Bogotá</span>
                </div>
                <div className="h-8 w-px bg-gray-300 mx-2"></div>
                <div className="flex items-center">
                    <span className="font-bold text-gray-700 text-lg tracking-tight">stratesys</span>
                </div>
            </div>
            <div className="hidden md:block text-xs text-gray-400 font-medium">
                PROYECTO MEGA - III ENCUENTRO
            </div>
        </header>
    );

    // Progress Bar
    const ProgressBar = () => (
        <div className="w-full bg-gray-200 h-2 mt-8 mb-8 rounded-full overflow-hidden">
            <div 
                className="bg-[#E30613] h-full transition-all duration-500 ease-out"
                style={{ width: `${((step - 1) / 3) * 100}%` }}
            ></div>
        </div>
    );

    // Renderizado del mensaje de éxito/error
    const renderSubmitMessage = () => {
        if (submitStatus === 'success') {
            return (
                <div className="text-center py-10 fade-in flex flex-col items-center justify-center h-full">
                    <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mb-6">
                        <CheckCircle2 size={40} className="text-green-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-800 mb-3">¡Gracias por participar!</h2>
                    <p className="text-gray-500 max-w-md mx-auto mb-8">
                        Tu respuesta ha sido registrada correctamente en nuestra base de datos.
                    </p>
                    <button 
                        onClick={resetForm}
                        className="text-sm font-medium text-[#E30613] hover:text-[#BA0C2F] underline decoration-2 underline-offset-4 transition-colors"
                    >
                        Volver al inicio
                    </button>
                </div>
            );
        } else if (submitStatus === 'error') {
            return (
                <div className="text-center py-10 fade-in flex flex-col items-center justify-center h-full">
                    <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mb-6">
                        <Cpu size={40} className="text-red-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-800 mb-3">¡Ocurrió un error!</h2>
                    <p className="text-gray-500 max-w-md mx-auto mb-8">
                        No pudimos registrar tu respuesta. **Por favor, verifica dos cosas:** 1) Que la clave (`supabaseAnonKey`) sea la clave pública de tu proyecto. 2) Que la tabla `respuestas` tenga una **Regla de Seguridad a Nivel de Fila (RLS)** habilitada para permitir `INSERT` para el rol `anon`.
                    </p>
                    <button 
                        onClick={() => setStep(1)}
                        className="text-sm font-medium text-[#E30613] hover:text-[#BA0C2F] underline decoration-2 underline-offset-4 transition-colors"
                    >
                        Reintentar
                    </button>
                </div>
            );
        }
        return null; // No debería suceder si submitStatus es null y step es 4
    };
    
    // Indicador de carga inicial
    if (isClientLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-[#F9F9F9] font-sans">
                <div className="flex flex-col items-center">
                    <svg className="animate-spin h-8 w-8 text-[#E30613] mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p className="text-gray-600">Cargando servicios de datos...</p>
                </div>
            </div>
        );
    }


    return (
        <div className="min-h-screen bg-[#F9F9F9] font-sans text-gray-900 pb-12">
            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
                
                body {
                    font-family: 'Inter', sans-serif;
                }
                
                .fade-in {
                    animation: fadeIn 0.5s ease-in-out;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                ::-webkit-scrollbar {
                    width: 8px;
                }
                ::-webkit-scrollbar-track {
                    background: #f1f1f1; 
                }
                ::-webkit-scrollbar-thumb {
                    background: #ccc; 
                    border-radius: 4px;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #E30613; 
                }
            `}</style>

            <Header />

            <main className="pt-24 px-4 md:px-0 max-w-3xl mx-auto">
                
                {step < 4 && (
                    <div className="text-center mb-10 fade-in">
                        <h1 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">
                            Construyamos una visión conjunta
                        </h1>
                        <p className="text-gray-500 text-sm md:text-base">
                            Encuesta Proyecto MEGA – Tu participación construye el futuro empresarial de Bogotá.
                        </p>
                        <ProgressBar />
                    </div>
                )}

                <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6 md:p-10 min-h-[400px] relative overflow-hidden transition-all">
                    
                    {/* PASO 1: INFORMACIÓN GENERAL */}
                    {step === 1 && (
                        <div className="fade-in space-y-8">
                            <div className="flex items-center gap-2 mb-6 border-b pb-4">
                                <Building2 className="text-[#E30613]" size={24} />
                                <h2 className="text-xl font-semibold text-gray-800">Información General</h2>
                            </div>

                            {/* Industria */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    ¿A qué industria pertenece su empresa? <span className="text-red-500">*</span>
                                </label>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {industries.map((ind) => (
                                        <button
                                            key={ind}
                                            onClick={() => handleInputChange('industry', ind)}
                                            className={`px-4 py-3 rounded-lg text-left text-sm transition-all border ${
                                                formData.industry === ind
                                                    ? 'bg-red-50 border-[#E30613] text-[#E30613] font-medium shadow-sm'
                                                    : 'bg-white border-gray-200 text-gray-600 hover:border-gray-300 hover:bg-gray-50'
                                            }`}
                                        >
                                            {ind}
                                        </button>
                                    ))}
                                </div>
                                {errors.industry && <p className="text-xs text-red-500 mt-1">{errors.industry}</p>}
                            </div>

                            {/* Ventas */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                                    <DollarSign size={16} /> Volumen anual de ventas <span className="text-red-500">*</span>
                                </label>
                                <select 
                                    value={formData.salesVolume}
                                    onChange={(e) => handleInputChange('salesVolume', e.target.value)}
                                    className={`w-full p-3 rounded-lg border focus:ring-2 focus:ring-red-100 focus:border-[#E30613] outline-none transition-all ${errors.salesVolume ? 'border-red-500' : 'border-gray-300'}`}
                                >
                                    <option value="">Seleccione una opción</option>
                                    {salesVolumes.map(vol => <option key={vol} value={vol}>{vol}</option>)}
                                </select>
                                {errors.salesVolume && <p className="text-xs text-red-500 mt-1">{errors.salesVolume}</p>}
                            </div>

                            {/* Empleados */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                                    <Users size={16} /> Número de empleados <span className="text-red-500">*</span>
                                </label>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                    {employeeRanges.map((range) => (
                                        <div 
                                            key={range}
                                            onClick={() => handleInputChange('employees', range)}
                                            className={`cursor-pointer p-3 rounded-lg border flex items-center gap-3 transition-all ${
                                                formData.employees === range
                                                    ? 'bg-red-50 border-[#E30613]'
                                                    : 'bg-white border-gray-200 hover:bg-gray-50'
                                            }`}
                                        >
                                            <div className={`w-4 h-4 rounded-full border flex items-center justify-center ${
                                                formData.employees === range ? 'border-[#E30613]' : 'border-gray-400'
                                            }`}>
                                                {formData.employees === range && <div className="w-2 h-2 rounded-full bg-[#E30613]"></div>}
                                            </div>
                                            <span className={`text-sm ${formData.employees === range ? 'text-[#E30613] font-medium' : 'text-gray-600'}`}>{range}</span>
                                        </div>
                                    ))}
                                </div>
                                {errors.employees && <p className="text-xs text-red-500 mt-1">{errors.employees}</p>}
                            </div>
                        </div>
                    )}

                    {/* PASO 2: ADOPCIÓN TECNOLÓGICA */}
                    {step === 2 && (
                        <div className="fade-in space-y-8">
                            <div className="flex items-center gap-2 mb-6 border-b pb-4">
                                <TrendingUp className="text-[#E30613]" size={24} />
                                <h2 className="text-xl font-semibold text-gray-800">Nivel de Adopción Tecnológica</h2>
                            </div>

                            {/* Nivel Adopción */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-3">
                                    ¿Cuál es el nivel de adopción tecnológica de su empresa? <span className="text-red-500">*</span>
                                </label>
                                <div className="space-y-3">
                                    {techLevels.map((level) => (
                                        <div 
                                            key={level.id}
                                            onClick={() => handleInputChange('techAdoption', level.id)}
                                            className={`cursor-pointer p-4 rounded-xl border transition-all flex items-center justify-between ${
                                                formData.techAdoption === level.id
                                                    ? `${level.bg} ${level.border} ring-1 ring-offset-1 ring-gray-200 shadow-sm`
                                                    : 'bg-white border-gray-200 hover:border-gray-300'
                                            }`}
                                        >
                                            <div className="flex items-center gap-3">
                                                <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                                                    formData.techAdoption === level.id ? `border-${level.color.split('-')[1]}-600` : 'border-gray-300'
                                                }`}>
                                                    {formData.techAdoption === level.id && <div className={`w-2.5 h-2.5 rounded-full bg-${level.color.split('-')[1]}-600`}></div>}
                                                </div>
                                                <span className={`text-sm font-medium ${formData.techAdoption === level.id ? 'text-gray-900' : 'text-gray-600'}`}>
                                                    {level.label}
                                                </span>
                                            </div>
                                            {formData.techAdoption === level.id && <CheckCircle2 size={18} className={level.color} />}
                                        </div>
                                    ))}
                                </div>
                                {errors.techAdoption && <p className="text-xs text-red-500 mt-1">{errors.techAdoption}</p>}
                            </div>

                            {/* Tecnologías Usadas */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
                                    <Cpu size={16} /> ¿Qué tecnologías utiliza actualmente?
                                </label>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {techList.map((tech) => (
                                        <label 
                                            key={tech} 
                                            className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all select-none ${
                                                formData.technologies.includes(tech)
                                                    ? 'bg-gray-50 border-gray-400'
                                                    : 'bg-white border-gray-200 hover:bg-gray-50'
                                            }`}
                                        >
                                            <div className={`w-5 h-5 rounded border flex items-center justify-center transition-colors ${
                                                formData.technologies.includes(tech) ? 'bg-[#E30613] border-[#E30613]' : 'bg-white border-gray-300'
                                            }`}>
                                                {formData.technologies.includes(tech) && <Check size={14} color="white" />}
                                            </div>
                                            <input 
                                                type="checkbox" 
                                                className="hidden" 
                                                checked={formData.technologies.includes(tech)}
                                                onChange={() => handleCheckboxChange(tech)}
                                            />
                                            <span className="text-sm text-gray-700">{tech}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* PASO 3: PROPÓSITO Y VISIÓN */}
                    {step === 3 && (
                        <div className="fade-in space-y-8">
                            <div className="flex items-center gap-2 mb-6 border-b pb-4">
                                <Lightbulb className="text-[#E30613]" size={24} />
                                <h2 className="text-xl font-semibold text-gray-800">Propósito y Visión</h2>
                            </div>

                            {/* Propósito Actual */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                                    <Target size={16} /> ¿Cual es el propósito actual de su empresa? <span className="text-red-500">*</span>
                                </label>
                                <textarea
                                    value={formData.purposeCurrent}
                                    onChange={(e) => handleInputChange('purposeCurrent', e.target.value)}
                                    maxLength={300}
                                    placeholder="Describa brevemente el propósito misión o razón de ser actual de su empresa."
                                    className={`w-full p-4 rounded-xl border h-32 resize-none focus:ring-2 focus:ring-red-100 focus:border-[#E30613] outline-none transition-all ${
                                        errors.purposeCurrent ? 'border-red-500' : 'border-gray-300'
                                    }`}
                                ></textarea>
                                <div className="flex justify-between mt-1">
                                    {errors.purposeCurrent ? <span className="text-xs text-red-500">{errors.purposeCurrent}</span> : <span></span>}
                                    <span className={`text-xs ${formData.purposeCurrent.length === 300 ? 'text-red-600 font-bold' : 'text-gray-400'}`}>
                                        {formData.purposeCurrent.length} / 300
                                    </span>
                                </div>
                            </div>

                            {/* Propósito Futuro */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                                    <BarChart3 size={16} /> ¿Cual considera será el propósito de su empresa en 18 años? <span className="text-red-500">*</span>
                                </label>
                                <textarea
                                    value={formData.purposeFuture}
                                    onChange={(e) => handleInputChange('purposeFuture', e.target.value)}
                                    maxLength={300}
                                    placeholder="Describa cómo imagina la evolución y propósito futuro de su empresa en 18 años."
                                    className={`w-full p-4 rounded-xl border h-32 resize-none focus:ring-2 focus:ring-red-100 focus:border-[#E30613] outline-none transition-all ${
                                        errors.purposeFuture ? 'border-red-500' : 'border-gray-300'
                                    }`}
                                ></textarea>
                                <div className="flex justify-between mt-1">
                                    {errors.purposeFuture ? <span className="text-xs text-red-500">{errors.purposeFuture}</span> : <span></span>}
                                    <span className={`text-xs ${formData.purposeFuture.length === 300 ? 'text-red-600 font-bold' : 'text-gray-400'}`}>
                                        {formData.purposeFuture.length} / 300
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* PASO 4: ÉXITO/ERROR */}
                    {step === 4 && renderSubmitMessage()}

                    {/* Botones de Navegación */}
                    {step < 4 && (
                        <div className="flex justify-between items-center mt-10 pt-6 border-t border-gray-100">
                            {step > 1 ? (
                                <button 
                                    onClick={prevStep}
                                    className="flex items-center gap-2 text-gray-500 font-medium hover:text-gray-700 px-4 py-2 rounded-lg transition-colors"
                                >
                                    <ArrowLeft size={18} /> Atrás
                                </button>
                            ) : (
                                <div></div> // Spacer
                            )}

                            {step < 3 ? (
                                <button 
                                    onClick={nextStep}
                                    className="flex items-center gap-2 bg-[#E30613] hover:bg-[#BA0C2F] text-white font-medium px-8 py-3 rounded-full shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-0.5"
                                >
                                    Siguiente <ArrowRight size={18} />
                                </button>
                            ) : (
                                <button 
                                    onClick={handleSubmit}
                                    disabled={isSubmitting || isClientLoading}
                                    className="flex items-center gap-2 bg-[#E30613] hover:bg-[#BA0C2F] text-white font-medium px-10 py-3 rounded-full shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-0.5 disabled:opacity-70 disabled:cursor-not-allowed"
                                >
                                    {isSubmitting || isClientLoading ? (
                                        <span className="flex items-center gap-2">
                                            <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                            </svg>
                                            {isClientLoading ? 'Cargando...' : 'Enviando...'}
                                        </span>
                                    ) : (
                                        <>Enviar <CheckCircle2 size={18} /></>
                                    )}
                                </button>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer minimalista */}
                <div className="text-center mt-8 pb-8 text-gray-400 text-xs">
                    © 2025 Cámara de Comercio de Bogotá & Stratesys
                </div>

            </main>
        </div>
    );
}