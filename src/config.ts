export const SITE = {
  website: "https://fredchyan.github.io/",
  author: "Fred Chyan",
  profile: "https://fredchyan.github.io/",
  desc: "Fred Chyan's personal blog - learning and sharing in public.",
  title: "Fred Chyan",
  ogImage: "fredchyan-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false, // disable edit links for now
    text: "Edit page",
    url: "https://github.com/fredchyan/fredchyan/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "America/Los_Angeles",
} as const;