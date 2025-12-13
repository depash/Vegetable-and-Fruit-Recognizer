import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css"; // <--- Correctly importing the CSS here

// [FONT DEFINITIONS REMAIN HERE]

export const metadata: Metadata = {
    // [METADATA REMAINS HERE]
};

// ------------------------------------------------------------------
// --- REMOVE THIS SECTION FROM LAYOUT.TSX:
/*
* / ** @type {import('tailwindcss').Config} * /
* module.exports = {
* content: [ ... ],
* theme: { ... },
* plugins: [],
* };
*/
// ------------------------------------------------------------------

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    // [RETURN STATEMENT REMAINS HERE]
}