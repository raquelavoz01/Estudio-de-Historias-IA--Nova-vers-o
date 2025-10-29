// FIX: Removed `/// <reference types="vite/client" />` which was causing a "Cannot find type definition file" error.
// This is no longer needed as the project now uses process.env.

export interface MagicTool {
  id: string;
  name: string;
  description: string;
  icon: string; // emoji
  prompt: string;
}

export interface Character {
  name: string;
  description: string;
}

export interface Chapter {
  title: string;
  outline: string;
  content: string;
  audio: string;
  isNarrating: boolean;
  narrationStyle: string;
}

export interface Story {
  id: string;
  premise: string;
  title: string;
  genre: string;
  summary: string;
  authorName: string;
  characters: Character[];
  chapters: Chapter[];
  coverImage: string; // Base64 image string
  synopsis: string;
  description: string;
  tags: string[];
  userCoverImage: string;
  userCoverImageMimeType: string;
  chapterCount: number;
  wordCount: number;
}

export type LoadingState = 'concept' | 'characters' | 'chapters' | 'content' | 'audio' | 'synopsis' | 'cover' | 'description' | 'tags' | 'realistic_cover' | 'tool_analysis' | 'tool_execution' | null;

export type LoadingTarget = {
    type: LoadingState;
    index?: number;
};

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  content: string | { imageUrl: string; alt: string };
}
