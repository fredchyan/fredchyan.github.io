import type { CollectionEntry } from "astro:content";
import { slugifyStr } from "./slugify";
import postFilter from "./postFilter";

interface Tag {
  tag: string;        // URL-safe slug: "javascript-tips"
  tagName: string;    // Original display name: "JavaScript Tips"
}

/**
 * Extracts all unique tags from blog posts and returns them as objects
 * with both URL-safe slugs and original display names.
 * 
 * @param posts - Array of blog post entries
 * @returns Array of unique tag objects sorted alphabetically by slug
 */
const getUniqueTags = (posts: CollectionEntry<"blog">[]) => {
  const tags: Tag[] = posts
    // Step 1: Filter out draft posts and future scheduled posts
    .filter(postFilter)

    // Step 2: Flatten all tags from all posts into a single array
    // [[tag1, tag2], [tag3, tag1]] → [tag1, tag2, tag3, tag1]
    .flatMap(post => post.data.tags)

    // Step 3: Convert each tag string to an object with both slug and display name
    // "JavaScript Tips" → { tag: "javascript-tips", tagName: "JavaScript Tips" }
    .map(tag => ({ tag: slugifyStr(tag), tagName: tag }))

    // Step 4: Remove duplicates based on the URL-safe slug
    // If "JavaScript Tips" and "javascript tips" both exist, keep only the first one
    // How it works:
    // - For each tag object, find the FIRST index where this slug appears in the array
    // - If the current index equals that first index, keep it (it's the first occurrence)
    // - If the current index is different, discard it (it's a duplicate)
    // Example: ["js-tips", "react", "js-tips"] → keep index 0, discard index 2
    .filter(
      (value, index, self) =>
        self.findIndex(tag => tag.tag === value.tag) === index
    )

    // Step 5: Sort alphabetically by the URL-safe slug
    .sort((tagA, tagB) => tagA.tag.localeCompare(tagB.tag));

  return tags;
};

export default getUniqueTags;