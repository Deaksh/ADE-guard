import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0b1f2a",
        sky: "#e4f1f8",
        sea: "#1d6f8a",
        mint: "#59d3b8",
        sand: "#f5efe6",
        coral: "#f26b5b",
        slate: "#4b5563"
      },
      fontFamily: {
        display: ["'Space Grotesk'", "ui-sans-serif", "system-ui"],
        body: ["'IBM Plex Sans'", "ui-sans-serif", "system-ui"]
      },
      boxShadow: {
        soft: "0 20px 50px -25px rgba(11, 31, 42, 0.35)",
      }
    },
  },
  plugins: [],
};

export default config;
