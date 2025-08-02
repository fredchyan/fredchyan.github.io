// @ts-check
import { defineConfig, envField } from 'astro/config';

import sitemap from '@astrojs/sitemap';
import react from '@astrojs/react';
import remarkToc from 'remark-toc';
import remarkCollapse from 'remark-collapse';

// ===== LaTeX Math Rendering Pipeline =====
// The complete pipeline for processing LaTeX math in markdown:
// 
// Markdown with LaTeX → remark-math → rehype-katex → katex → Rendered Math
//
// remark-math (Markdown → AST)
// - Parses LaTeX syntax: $E = mc^2$ (inline) and $$\int f(x)dx$$ (block)
// - Converts LaTeX strings into math AST nodes in the markdown syntax tree
// - Does NOT render math, just identifies and structures it
import remarkMath from 'remark-math';

// rehype-katex (AST → HTML)
// - Takes math AST nodes and renders them to HTML using KaTeX
// - Produces <span class="katex"> elements with SVG/HTML math content
// - KaTeX is faster than MathJax and provides high-quality math rendering
import rehypeKatex from 'rehype-katex';

import tailwindcss from '@tailwindcss/vite';
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
} from "@shikijs/transformers";
import { transformerFileName } from "./src/utils/transformers/fileName.js";
import { SITE } from "./src/config.ts";

// https://astro.build/config
export default defineConfig({
  site: SITE.website,
  integrations: [
    react(),
    sitemap({
      filter: page => SITE.showArchives || !page.endsWith("/archives"),
    }),
  ],

  markdown: {
    remarkPlugins: [
      // ===== LaTeX Pipeline: Parse Math Syntax =====
      // remark-math identifies LaTeX delimiters and creates math AST nodes
      // Input:  "The formula $E = mc^2$ represents energy"
      // Output: AST with text node + math node + text node
      remarkMath,
      
      remarkToc,
      [
        remarkCollapse,
        {
          test: 'Table of contents',
        },
      ],
    ],
    
    // ===== LaTeX Pipeline: Render Math to HTML =====
    // rehype-katex takes math AST nodes and renders them using KaTeX
    // Input:  Math AST node with LaTeX string "E = mc^2"
    // Output: <span class="katex"><span class="katex-mathml">...</span></span>
    // Note: Requires katex.min.css (loaded in Layout.astro) for proper styling
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: { light: "min-light", dark: "night-owl" },
      defaultColor: false,
      wrap: false,
      transformers: [
        transformerFileName({ style: "v2", hideDot: false }),
        transformerNotationHighlight(),
        transformerNotationWordHighlight(),
        transformerNotationDiff({ matchAlgorithm: "v3" }),
      ],
    },
  },

  vite: {
    plugins: [tailwindcss()],
    optimizeDeps: {
      exclude: ["@resvg/resvg-js"],
    },
  },
  
  // Enhanced image optimization with responsive styles and constrained layout
  image: {
    responsiveStyles: true,
    layout: "constrained",
  },
  
  // Environment variable validation schema
  // PUBLIC_GOOGLE_SITE_VERIFICATION is optional - for Google Search Console integration
  env: {
    schema: {
      PUBLIC_GOOGLE_SITE_VERIFICATION: envField.string({
        access: "public",
        context: "client",
        optional: true,
      }),
    },
  },
  
  // Experimental features for better script loading order
  experimental: {
    preserveScriptOrder: true,
  },
});