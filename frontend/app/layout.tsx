import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ADEGuard | Real-time ADE Intelligence",
  description: "NLP-driven adverse drug event detection, clustering, and severity insights.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="surface min-h-screen">
        {children}
      </body>
    </html>
  );
}
