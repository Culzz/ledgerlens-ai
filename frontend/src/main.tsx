import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import SavedBrief from "./components/SavedBrief";
import "./styles.css";

const container = document.getElementById("root");
if (!container) throw new Error("no #root element in the page");

// Two pages share an origin: the app, and a brief someone was sent. The server
// serves index.html for both, so the path is what chooses between them.
const brief = /^\/b\/([A-Za-z0-9_-]{8,64})\/?$/.exec(window.location.pathname);

createRoot(container).render(
  <StrictMode>{brief ? <SavedBrief id={brief[1]} /> : <App />}</StrictMode>,
);
