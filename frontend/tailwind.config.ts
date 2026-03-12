import type { Config } from "tailwindcss";

export default {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
      },
      colors: {
        brand: {
          50:  "#eff6ff",
          100: "#dbeafe",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
        canvas:  "#f4f6f9",
        surface: "#ffffff",
        border:  "#e5e9ef",
        "text-primary":   "#0f1923",
        "text-secondary": "#4b5768",
        "text-muted":     "#8a96a8",
      },
      boxShadow: {
        card:       "0 1px 3px rgba(15,25,35,0.06), 0 1px 2px rgba(15,25,35,0.04)",
        "card-md":  "0 4px 16px rgba(15,25,35,0.10)",
        sidebar:    "1px 0 0 #e5e9ef",
      },
      borderRadius: {
        card: "12px",
      },
    },
  },
  plugins: [],
} satisfies Config;
