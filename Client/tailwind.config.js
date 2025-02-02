// tailwind.config.js
// eslint-disable-next-line no-undef
const { nextui } = require("@nextui-org/react");

/** @type {import('tailwindcss').Config} */
export default {
  mode: "jit",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./node_modules/@nextui-org/theme/dist/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  darkMode: "class",
  plugins: [
    nextui({
      addCommonColors: true,
      layout: {}, // common layout options
      colors: {
        primary: {
          DEFAULT: "#27374d",
        },
      },
      themes: {
        light: {},
        dark: {
          colors: {
            primary: {
              DEFAULT: "#27374d",
            },
          },
          layout: {}, // dark theme layout options
          // ...
        },
        // ... custom themes
      },
    }),
  ],
};
