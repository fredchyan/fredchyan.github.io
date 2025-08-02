import type { CollectionEntry } from "astro:content";

// Types for grouping functionality
type GroupKey = string | number | symbol;

/**
 * GroupFunction interface defines the shape of a function used for grouping
 * 
 * This is a TypeScript interface that describes what a grouping function should look like:
 * - It takes a generic type T (in our case, a blog post)
 * - It optionally takes an index number (the position in the array)
 * - It returns a GroupKey (string, number, or symbol) that identifies which group the item belongs to
 * 
 * Example implementations:
 * - (post) => post.data.pubDatetime.getFullYear()        // Groups by year (returns number like 2022)
 * - (post) => post.data.pubDatetime.getMonth() + 1      // Groups by month (returns number 1-12)
 * - (post) => post.data.tags[0]                         // Groups by first tag (returns string)
 * - (post, index) => index % 2 === 0 ? "even" : "odd"  // Groups by even/odd position
 */
interface GroupFunction<T> {
  (item: T, index?: number): GroupKey;
}

/**
 * Groups blog posts based on a custom condition function
 * 
 * @param posts - Array of blog posts to group
 * @param groupFunction - Function that determines the grouping key for each post
 * @returns Object where keys are group identifiers and values are arrays of posts
 * 
 * How it works:
 * 1. Takes an array of posts and a function that decides how to group them
 * 2. For each post, calls the groupFunction to get a "key" (like "2022" for year)
 * 3. Puts all posts with the same key into the same group
 * 4. Returns an object like: { "2022": [post1, post2], "2023": [post3, post4] }
 * 
 * Example usage in archives page:
 * - Group by year: getPostsByGroupCondition(posts, post => post.data.pubDatetime.getFullYear())
 *   Result: { "2022": [posts from 2022], "2023": [posts from 2023] }
 * - Group by month: getPostsByGroupCondition(posts, post => post.data.pubDatetime.getMonth() + 1)
 *   Result: { "1": [January posts], "2": [February posts], ... }
 */
const getPostsByGroupCondition = (
  posts: CollectionEntry<"blog">[],
  groupFunction: GroupFunction<CollectionEntry<"blog">>
) => {
  // Create an object to store grouped results
  // This will look like: { "2022": [post1, post2], "2023": [post3] }
  const result: Record<GroupKey, CollectionEntry<"blog">[]> = {};
  
  // Loop through each post
  for (let i = 0; i < posts.length; i++) {
    const item = posts[i];
    
    // Use the groupFunction to determine which group this post belongs to
    // For example, if groupFunction is (post) => post.data.pubDatetime.getFullYear()
    // then groupKey might be 2022, 2023, etc.
    const groupKey = groupFunction(item, i);
    
    // If this group doesn't exist yet, create it as an empty array
    if (!result[groupKey]) {
      result[groupKey] = [];
    }
    
    // Add the post to the appropriate group
    result[groupKey].push(item);
  }
  
  return result;
};

export default getPostsByGroupCondition;