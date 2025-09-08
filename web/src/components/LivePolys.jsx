import React, { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || "http://127.0.0.1:5001";

// Minimal preset map so /scene works
const SCENES = {
  calm:   { stroke:"#00e5ff", fill:"rgba(0,229,255,0.18)", lw:3, glow:6,  blend:"screen" },
  forest: { stroke:"#7bd389", fill:"rgba(123,211,137,0.22)", lw:4, glow:10, blend:"multiply" },
  neon:   { stroke:"#ff006e", fill:"rgba(255,0,110,0.18)", lw:5, glow:20, blend:"lighter" },
  flame:  { stroke:"#ffd166", fill:"rgba(255,209,102,0.20)", lw:6, glow:24, blend:"screen" },
  aurora: { stroke:"#b692ff", fill:"rgba(182,146,255,0.20)", lw:4, glow:16, blend:"screen" },
};

export default function LivePolys() {
  const canvasRef = useRef(null);
  const [polys, setPolys] = useState([]);

  // NEW: style state (defaults to calm)
  const [style, setStyle] = useState(SCENES.calm);

  // connect once
  useEffect(() => {
    const socket = io(SOCKET_URL, { transports: ["websocket"] });
    socket.on("connect", () => console.log("Connected to", SOCKET_URL));
    socket.on("mask",  (msg) => setPolys(msg?.polys || []));
    socket.on("style", (msg) => {
      console.log("style msg:", msg);
      setStyle((s) => ({ ...s, ...msg }));
    });
    socket.on("scene", (msg) => {
      console.log("scene msg:", msg);
      if (msg?.name && SCENES[msg.name]) setStyle(SCENES[msg.name]);
    });
    return () => socket.disconnect();
  }, []);

  // size canvas to device pixels
  useEffect(() => {
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    const resize = () => {
      const w = window.innerWidth, h = window.innerHeight;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  // draw polygons (now uses style coming from backend)
  useEffect(() => {
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    const { width, height } = c;

    ctx.clearRect(0, 0, width, height);

    // apply style
    const dpr = window.devicePixelRatio || 1;
    ctx.globalCompositeOperation = style.blend || "source-over";
    ctx.lineWidth   = (style.lw ?? 2) * dpr;
    ctx.strokeStyle = style.stroke || "rgba(0,255,0,1)";
    ctx.fillStyle   = style.fill   || "rgba(0,255,0,0.25)";
    ctx.shadowColor = style.glow ? (style.stroke || "#00ff00") : "transparent";
    ctx.shadowBlur  = style.glow || 0;

    polys.forEach((poly) => {
      if (!poly || poly.length < 3) return;
      ctx.beginPath();
      for (let i = 0; i < poly.length; i++) {
        const x = poly[i][0] * width;
        const y = poly[i][1] * height;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });
  }, [polys, style]);

  return (
    <canvas
      ref={canvasRef}
      style={{ display: "block", width: "100vw", height: "100vh", background: "black" }}
    />
  );
}
