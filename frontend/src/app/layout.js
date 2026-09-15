import "./globals.css";

export const metadata = {
  title: "SportsVision",
  description: "Human Action Recognition in Sports Videos",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}