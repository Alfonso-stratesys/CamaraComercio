import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseKey = import.meta.env.VITE_SUPABASE_KEY

// Validación simple para avisar si faltan las keys
if (!supabaseUrl || !supabaseKey) {
  console.error("⚠️ Faltan las credenciales de Supabase en el archivo .env (asegúrate de que empiecen con VITE_)");
}

export const supabase = createClient(supabaseUrl, supabaseKey)