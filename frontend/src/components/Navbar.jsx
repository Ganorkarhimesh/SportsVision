"use client";

import Link from "next/link";
import { Activity } from "lucide-react";

export default function Navbar() {
  return (
    <nav className="navbar">
      <Link href="/" className="logo">
        <Activity size={24} />
        <span>SportsVision</span>
      </Link>

      <div className="nav-links">
        <Link href="/">Home</Link>
        <Link href="/about">About</Link>

        <a
          href="https://github.com/SankalpBankar/SportsVision.git"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a>
      </div>
    </nav>
  );
}