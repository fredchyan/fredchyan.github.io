/**
 * Get full path of a blog post
 * Simplified version for flat blog structure
 * @param id - id of the blog post (already without .md extension)
 * @param filePath - the blog post full file location (unused in simple version)
 * @param includeBase - whether to include `/posts` in return value
 * @returns blog post path
 */
export function getPath(
  id: string,
  filePath?: string,
  includeBase = true
) {
  const basePath = includeBase ? "/posts" : "";
  return `${basePath}/${id}`;
}