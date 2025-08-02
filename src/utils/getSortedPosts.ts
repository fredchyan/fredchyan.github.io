import type { CollectionEntry } from "astro:content";
import postFilter from "./postFilter";

/**
 * Sorts blog posts by date (newest first) with smart date handling.
 * Uses modified date if available, falls back to publish date.
 * 
 * @param posts - Array of blog post entries
 * @returns Posts sorted by date (most recent first)
 */
const getSortedPosts = (posts: CollectionEntry<"blog">[]) => {
  return posts
    // Filter out draft posts and future scheduled posts
    .filter(postFilter)
    
    // Sort by date (newest first)
    .sort(
      (a, b) =>
        // Use modDatetime if available, otherwise use pubDatetime
        // Convert to Unix timestamp (seconds) for reliable comparison
        Math.floor(
          new Date(b.data.modDatetime ?? b.data.pubDatetime).getTime() / 1000
        ) -
        Math.floor(
          new Date(a.data.modDatetime ?? a.data.pubDatetime).getTime() / 1000
        )
    );
};

export default getSortedPosts;