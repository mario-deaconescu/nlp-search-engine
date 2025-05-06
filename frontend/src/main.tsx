import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import {HeroUIProvider, ToastProvider} from "@heroui/react";

createRoot(document.getElementById('root')!).render(
  <StrictMode>
      <HeroUIProvider>
        <ToastProvider
        placement={"top-center"}
        toastOffset={50}
        />
          <App />
      </HeroUIProvider>
  </StrictMode>,
)
