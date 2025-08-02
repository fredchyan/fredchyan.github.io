// @ts-check
import { defineConfig, envField } from 'astro/config';

import sitemap from '@astrojs/sitemap';
import remarkToc from 'remark-toc';
import remarkCollapse from 'remark-collapse';

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
    sitemap({
      filter: page => SITE.showArchives || !page.endsWith("/archives"),
    }),
  ],

  markdown: {
    remarkPlugins: [
      remarkToc,
      [
        remarkCollapse,
        {
          test: 'Table of contents',
        },
      ],
    ],
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