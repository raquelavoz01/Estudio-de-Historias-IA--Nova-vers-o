import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { Story, LoadingTarget, ChatMessage, Character, MagicTool } from './types';
import { GoogleGenAI, Chat } from '@google/genai';
import { generateConcept, generateChapterOutlines, generateChapterContent, narrateText, generateSynopsis, generateCoverImage, generateConceptArt, generateDescription, generateTags, generateRealisticCover, analyzeToolPrompt, executeMagicTool } from './services/geminiService';

// --- Helper Functions & Components ---

const decode = (base64: string): Uint8Array => {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
};

const decodeAudioData = async (data: Uint8Array, ctx: AudioContext): Promise<AudioBuffer> => {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length;
    const buffer = ctx.createBuffer(1, frameCount, 24000);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < frameCount; i++) {
        channelData[i] = dataInt16[i] / 32768.0;
    }
    return buffer;
};

const createWavBlob = (pcmData: Uint8Array, sampleRate: number, numChannels: number, bitsPerSample: number): Blob => {
    const dataSize = pcmData.length;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    const writeString = (offset: number, str: string) => {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    };

    const blockAlign = numChannels * (bitsPerSample / 8);
    const byteRate = sampleRate * blockAlign;

    // RIFF header
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');

    // fmt sub-chunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Sub-chunk size
    view.setUint16(20, 1, true); // Audio format (1 for PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // data sub-chunk
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    // Write PCM data
    const pcmAsDataView = new DataView(pcmData.buffer);
    for (let i = 0; i < dataSize; i++) {
        view.setUint8(44 + i, pcmAsDataView.getUint8(i));
    }
    
    return new Blob([view], { type: 'audio/wav' });
};


const Spinner: React.FC = () => (
    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);

const SaveStatusIndicator: React.FC<{ status: 'idle' | 'saving' | 'saved' }> = ({ status }) => {
    if (status === 'saving') {
        return (
            <div className="flex items-center gap-2 text-sm text-gray-400 animate-fade-in">
                <Spinner /> Salvando...
            </div>
        );
    }
    if (status === 'saved') {
        return (
            <div className="flex items-center gap-2 text-sm text-green-400 animate-fade-in">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm3.857-9.809a.75.75 0 0 0-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 1 0-1.06 1.061l2.5 2.5a.75.75 0 0 0 1.137-.089l4-5.5Z" clipRule="evenodd" /></svg>
                Salvo no seu navegador
            </div>
        );
    }
    return <div className="h-5"></div>; // Placeholder to prevent layout shift
};


const PlayIcon: React.FC<{className?: string}> = ({className}) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className || "w-6 h-6"}>
        <path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.647c1.295.742 1.295 2.545 0 3.286L7.279 20.99c-1.25.717-2.779-.217-2.779-1.643V5.653Z" clipRule="evenodd" />
    </svg>
);

const PauseIcon: React.FC<{className?: string}> = ({className}) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className={className || "w-6 h-6"}>
        <path fillRule="evenodd" d="M6.75 5.25a.75.75 0 0 1 .75.75v12a.75.75 0 0 1-1.5 0v-12a.75.75 0 0 1 .75-.75Zm9 0a.75.75 0 0 1 .75.75v12a.75.75 0 0 1-1.5 0v-12a.75.75 0 0 1 .75-.75Z" clipRule="evenodd" />
    </svg>
);

interface AudioPlayerProps {
    audioBase64: string;
    onPlaybackChange: (isPlaying: boolean) => void;
}

const AudioPlayer: React.FC<AudioPlayerProps> = ({ audioBase64, onPlaybackChange }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const audioContextRef = useRef<AudioContext | null>(null);
    const audioBufferRef = useRef<AudioBuffer | null>(null);
    const sourceRef = useRef<AudioBufferSourceNode | null>(null);

    const togglePlay = async () => {
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }

        if (isPlaying) {
            sourceRef.current?.stop();
            setIsPlaying(false);
            onPlaybackChange(false);
        } else {
            if (!audioBufferRef.current) {
                const decodedData = decode(audioBase64);
                audioBufferRef.current = await decodeAudioData(decodedData, audioContextRef.current);
            }

            const source = audioContextRef.current.createBufferSource();
            source.buffer = audioBufferRef.current;
            source.connect(audioContextRef.current.destination);
            source.onended = () => {
                setIsPlaying(false);
                onPlaybackChange(false);
            };
            source.start();
            sourceRef.current = source;
            setIsPlaying(true);
            onPlaybackChange(true);
        }
    };

    return (
        <button
            onClick={togglePlay}
            className="flex items-center justify-center p-2 rounded-full bg-indigo-500 hover:bg-indigo-400 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500"
            aria-label={isPlaying ? "Pausar narração" : "Ouvir narração"}
        >
            {isPlaying ? <PauseIcon className="w-5 h-5 text-white" /> : <PlayIcon className="w-5 h-5 text-white" />}
        </button>
    );
};

const initialChatMessages: ChatMessage[] = [
    {
      id: crypto.randomUUID(),
      role: 'model',
      content: "Olá! Sou Kekakekinhafan, seu assistente criativo. Como posso ajudar a dar vida à sua história hoje? Precisa de ideias para um personagem, um diálogo, um plot twist, ou talvez descrever um cenário?",
    },
];

const NARRATION_STYLES = ['Padrão', 'Triste', 'Alegre', 'Emocionado', 'Sentimental', 'Sensual', 'Sombrio', 'Épico', 'Sussurrante'];

// --- Main Application Component ---

export default function App() {
    const [stories, setStories] = useState<Story[]>([]);
    const [tools, setTools] = useState<MagicTool[]>([]);
    const [view, setView] = useState<'studio' | 'tools'>('studio');
    const [activeStoryId, setActiveStoryId] = useState<string | null>(null);
    const [loading, setLoading] = useState<LoadingTarget | null>(null);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    
    const [newCharName, setNewCharName] = useState('');
    const [newCharDesc, setNewCharDesc] = useState('');

    // Chat Assistant State
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [chatHistory, setChatHistory] = useState<ChatMessage[]>(initialChatMessages);
    const [chatInput, setChatInput] = useState('');
    const [isChatLoading, setIsChatLoading] = useState(false);
    const chatRef = useRef<Chat | null>(null);
    const chatBodyRef = useRef<HTMLDivElement>(null);
    
    // Save State
    const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');
    const saveTimeoutRef = useRef<number | null>(null);
    const isInitialRender = useRef(true);

    // Magic Tool State
    const [newToolPrompt, setNewToolPrompt] = useState('');
    const [activeTool, setActiveTool] = useState<MagicTool | null>(null);
    const [toolInput, setToolInput] = useState('');
    const [toolOutput, setToolOutput] = useState('');
    const [isToolRunning, setIsToolRunning] = useState(false);


    // Load data from local storage
    useEffect(() => {
        try {
            const savedData = localStorage.getItem('ia-studio-data');
            if (savedData) {
                const { stories: savedStories, tools: savedTools } = JSON.parse(savedData);
                if (Array.isArray(savedStories)) setStories(savedStories);
                if (Array.isArray(savedTools)) setTools(savedTools);
            } else {
                // Migration from old version
                const oldSavedStories = localStorage.getItem('ia-story-studio-stories');
                if (oldSavedStories) {
                    setStories(JSON.parse(oldSavedStories));
                    localStorage.removeItem('ia-story-studio-stories');
                }
            }
        } catch (e) {
            console.error("Failed to load data:", e);
            setError("Não foi possível carregar seus dados salvos.");
        }
    }, []);

    // Save data to local storage with debounce and visual feedback
    useEffect(() => {
        if (isInitialRender.current) {
            isInitialRender.current = false;
            return;
        }

        if (saveTimeoutRef.current) {
            clearTimeout(saveTimeoutRef.current);
        }
        setSaveStatus('saving');

        const timer = setTimeout(() => {
            try {
                const dataToSave = { stories, tools };
                localStorage.setItem('ia-studio-data', JSON.stringify(dataToSave));
                setSaveStatus('saved');

                saveTimeoutRef.current = window.setTimeout(() => {
                    setSaveStatus('idle');
                }, 2000);

            } catch (e) {
                console.error("Failed to save data:", e);
                setSaveStatus('idle');
            }
        }, 1000);

        return () => {
            clearTimeout(timer);
            if (saveTimeoutRef.current) {
                clearTimeout(saveTimeoutRef.current);
            }
        };
    }, [stories, tools]);
    
    useEffect(() => {
        if (chatBodyRef.current) {
            chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
        }
    }, [chatHistory]);

    const activeStory = stories.find(s => s.id === activeStoryId);

    const updateActiveStory = (updater: (story: Story) => Partial<Story>) => {
        setStories(prev => prev.map(s => s.id === activeStoryId ? { ...s, ...updater(s) } : s));
    };

    const handleAction = useCallback(async <T,>(
        loadingTarget: LoadingTarget,
        apiCall: () => Promise<T>,
        onSuccess: (result: T) => void
    ) => {
        setLoading(loadingTarget);
        setError(null);
        try {
            const result = await apiCall();
            onSuccess(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Ocorreu um erro desconhecido.');
        } finally {
            setLoading(null);
        }
    }, []);
    
    const handleCreateNewStory = () => {
        const newStory: Story = {
            id: crypto.randomUUID(), premise: '', title: '', genre: '', summary: '',
            authorName: '', characters: [], chapters: [], coverImage: '', synopsis: '', description: '',
            tags: [], userCoverImage: '', userCoverImageMimeType: '',
            chapterCount: 7, wordCount: 4000
        };
        setStories(prev => [newStory, ...prev]);
        setActiveStoryId(newStory.id);
    };

    const handleGenerateConcept = () => {
        if (!activeStory?.premise) return;
        handleAction(
            { type: 'concept' }, 
            () => generateConcept(activeStory.premise), 
            (result) => updateActiveStory(() => ({ 
                ...result,
                chapters: [], // Limpa capítulos antigos
                synopsis: '', // Limpa sinopse antiga
                description: '', // Limpa descrição antiga
                tags: [] // Limpa tags antigas
            }))
        );
    };

    const handleAddCharacter = (e: React.FormEvent) => {
        e.preventDefault();
        if (!newCharName.trim() || !newCharDesc.trim()) {
            setError("Nome e descrição do personagem são obrigatórios.");
            return;
        }
        const newCharacter: Character = { name: newCharName, description: newCharDesc };
        updateActiveStory(story => ({
            characters: [...story.characters, newCharacter]
        }));
        setNewCharName('');
        setNewCharDesc('');
    };

    const handleRemoveCharacter = (indexToRemove: number) => {
        updateActiveStory(story => ({
            characters: story.characters.filter((_, index) => index !== indexToRemove)
        }));
    };

    const handleGenerateChapters = () => {
        if (!activeStory?.summary || !activeStory?.characters || !activeStory?.chapterCount) return;
        handleAction({ type: 'chapters' }, () => generateChapterOutlines(activeStory.summary, activeStory.characters, activeStory.chapterCount), (outlines) => updateActiveStory(() => ({ chapters: outlines.map(o => ({...o, content: '', audio: '', isNarrating: false, narrationStyle: 'padrão'})) })));
    };

    const handleGenerateContent = (chapterIndex: number) => {
        if (!activeStory?.summary || !activeStory?.chapters || !activeStory?.wordCount || !activeStory?.characters) return;
        const previousChapters = activeStory.chapters.slice(0, chapterIndex);
        const recentChaptersContent = previousChapters.slice(-2).map((c, i) => `--- Conteúdo do Capítulo Anterior (${previousChapters.length - 1 + i}) ---\nTítulo: ${c.title}\n\n${c.content}\n--- Fim do Conteúdo ---`).join('\n\n');
        const allChapterOutlines = activeStory.chapters.map((c, i) => `${i + 1}. ${c.title}: ${c.outline}`).join('\n');
        const characterProfiles = activeStory.characters.map(c => `- ${c.name}: ${c.description}`).join('\n');
        const storyContext = `Resumo Geral da História: ${activeStory.summary}\n\nPersonagens Principais:\n${characterProfiles}\n\nEsboço de todos os capítulos:\n${allChapterOutlines}\n\nConteúdo recente para dar continuidade (últimos dois capítulos escritos):\n${recentChaptersContent || 'Nenhum capítulo anterior foi escrito ainda.'}`;
        handleAction(
            { type: 'content', index: chapterIndex }, 
            () => generateChapterContent(activeStory.summary, activeStory.chapters[chapterIndex], storyContext, activeStory.wordCount), 
            (content) => updateActiveStory(story => {
                const newChapters = [...story.chapters];
                newChapters[chapterIndex].content = content;
                return { chapters: newChapters };
            })
        );
    };

    const handleNarrationStyleChange = (chapterIndex: number, newStyle: string) => {
        updateActiveStory(story => {
            const newChapters = [...story.chapters];
            newChapters[chapterIndex].narrationStyle = newStyle;
            return { chapters: newChapters };
        });
    };

    const handleNarrate = (chapterIndex: number) => {
        if (!activeStory?.chapters?.[chapterIndex]?.content) return;
        const style = activeStory.chapters[chapterIndex].narrationStyle || 'padrão';
        handleAction({ type: 'audio', index: chapterIndex }, () => narrateText(activeStory.chapters[chapterIndex].content, style), (audio) => updateActiveStory(story => {
            const newChapters = [...story.chapters];
            newChapters[chapterIndex].audio = audio;
            return { chapters: newChapters };
        }));
    };
    
    const handleDownloadAudio = (chapterIndex: number) => {
        if (!activeStory?.chapters?.[chapterIndex]?.audio) return;
        const { audio: audioBase64, title } = activeStory.chapters[chapterIndex];
        const pcmData = decode(audioBase64);
        const wavBlob = createWavBlob(pcmData, 24000, 1, 16);
        const url = URL.createObjectURL(wavBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${activeStory.title || 'historia'} - ${title}.wav`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    const handleGenerateSynopsis = () => {
        if (!activeStory?.title || !activeStory?.summary) return;
        handleAction({ type: 'synopsis' }, () => generateSynopsis(activeStory.title, activeStory.summary), (synopsis) => updateActiveStory(() => ({ synopsis })));
    };
    
    const handleGenerateDescription = () => {
        if (!activeStory?.title || !activeStory?.summary) return;
        handleAction({ type: 'description' }, () => generateDescription(activeStory.title, activeStory.summary), (description) => updateActiveStory(() => ({ description })));
    };

    const handleGenerateTags = () => {
        if (!activeStory?.title || !activeStory?.summary || !activeStory?.genre) return;
        handleAction({ type: 'tags' }, () => generateTags(activeStory.title, activeStory.summary, activeStory.genre), (tags) => updateActiveStory(() => ({ tags })));
    };

    const handleGenerateCover = () => {
        if (!activeStory?.title || !activeStory?.summary) return;
        handleAction({ type: 'cover' }, () => generateCoverImage(activeStory.title, activeStory.summary, activeStory.authorName), (coverImage) => updateActiveStory(() => ({ coverImage })));
    };

    const handleGenerateRealisticCover = () => {
        if (!activeStory?.title || !activeStory?.summary || !activeStory.userCoverImage || !activeStory.userCoverImageMimeType) return;
        handleAction({ type: 'realistic_cover' }, () => generateRealisticCover(activeStory.title, activeStory.summary, activeStory.authorName, activeStory.userCoverImage, activeStory.userCoverImageMimeType), (coverImage) => updateActiveStory(() => ({ coverImage })));
    };

    const handleUserImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64String = (reader.result as string).replace(/^data:.+;base64,/, '');
            updateActiveStory(() => ({
                userCoverImage: base64String,
                userCoverImageMimeType: file.type
            }));
        };
        reader.readAsDataURL(file);
        event.target.value = '';
    };

    const handleDownloadCover = () => {
        if (!activeStory?.coverImage) return;
        const link = document.createElement('a');
        link.href = `data:image/jpeg;base64,${activeStory.coverImage}`;
        link.download = `${activeStory.title || 'capa'}.jpeg`;
        link.click();
    };

    const handlePlaybackChange = (chapterIndex: number, isPlaying: boolean) => {
        updateActiveStory(story => ({
            chapters: story.chapters.map((ch, idx) => ({ ...ch, isNarrating: idx === chapterIndex && isPlaying }))
        }));
    };
    
    const handleExport = () => {
        if (stories.length === 0 && tools.length === 0) return setError("Não há nada para exportar.");
        const dataStr = JSON.stringify({stories, tools}, null, 2);
        const link = document.createElement('a');
        link.href = URL.createObjectURL(new Blob([dataStr], { type: 'application/json' }));
        link.download = 'ia-studio-backup.json';
        link.click();
        URL.revokeObjectURL(link.href);
    };

    const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const imported = JSON.parse(e.target?.result as string);
                if (imported.stories && Array.isArray(imported.stories)) {
                    setStories(prev => {
                        const existingIds = new Set(prev.map(s => s.id));
                        return [...prev, ...imported.stories.filter((s: Story) => s.id && !existingIds.has(s.id))];
                    });
                }
                if (imported.tools && Array.isArray(imported.tools)) {
                    setTools(prev => {
                        const existingIds = new Set(prev.map(t => t.id));
                        return [...prev, ...imported.tools.filter((t: MagicTool) => t.id && !existingIds.has(t.id))];
                    });
                }
            } catch (err) {
                setError("Falha ao importar. Verifique o formato do arquivo.");
            }
        };
        reader.readAsText(file);
        event.target.value = '';
    };
    
    const openChat = () => {
      if (!chatRef.current) {
          // FIX: Switched from `import.meta.env.VITE_API_KEY` to `process.env.API_KEY` as per coding guidelines.
          const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
          chatRef.current = ai.chats.create({
              model: 'gemini-2.5-pro',
              config: {
                  systemInstruction: `Você é Kekakekinhafan, um assistente de IA criativo e amigável, especializado em ajudar escritores a desenvolver suas histórias. Você faz parte do 'Estúdio de Histórias IA'. Seja conciso, útil e inspirador. Responda em português. Se o usuário pedir para gerar uma imagem com um comando como '/imagem [prompt]', responda apenas com 'IMAGE_GENERATION_REQUEST:${'prompt'}' e nada mais.`,
              }
          });
      }
      setIsChatOpen(true);
    };

    const handleSendChatMessage = async (message: string) => {
        if (!message.trim() || isChatLoading) return;
        setIsChatLoading(true);
        setChatInput('');
        const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: message };
        setChatHistory(prev => [...prev, userMessage]);
        try {
            if (message.toLowerCase().startsWith('/imagem ')) {
                const prompt = message.substring(8);
                const imageUrl = await generateConceptArt(prompt);
                const imageMessage: ChatMessage = {
                    id: crypto.randomUUID(),
                    role: 'model',
                    content: { imageUrl: `data:image/jpeg;base64,${imageUrl}`, alt: `Arte conceitual para: ${prompt}` }
                };
                setChatHistory(prev => [...prev, imageMessage]);
            } else {
                if (!chatRef.current) openChat();
                const stream = await chatRef.current!.sendMessageStream({ message });
                let modelResponse = '';
                const modelMessageId = crypto.randomUUID();
                setChatHistory(prev => [...prev, { id: modelMessageId, role: 'model', content: '' }]);
                for await (const chunk of stream) {
                    modelResponse += chunk.text;
                    setChatHistory(prev => prev.map(msg => msg.id === modelMessageId ? { ...msg, content: modelResponse } : msg));
                }
            }
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Ocorreu um erro no chat.';
            setChatHistory(prev => [...prev, { id: crypto.randomUUID(), role: 'model', content: `Desculpe, ocorreu um erro: ${errorMsg}` }]);
        } finally {
            setIsChatLoading(false);
        }
    };

    const handleCreateTool = () => {
        if (!newToolPrompt.trim()) return;
        handleAction(
            { type: 'tool_analysis' },
            () => analyzeToolPrompt(newToolPrompt),
            (result) => {
                const newTool: MagicTool = {
                    id: crypto.randomUUID(),
                    prompt: newToolPrompt,
                    ...result
                };
                setTools(prev => [newTool, ...prev]);
                setNewToolPrompt('');
            }
        );
    };

    const handleExecuteTool = () => {
        if (!toolInput.trim() || !activeTool) return;
        setIsToolRunning(true);
        setToolOutput('');
        setError(null);
        executeMagicTool(activeTool.prompt, toolInput)
            .then(setToolOutput)
            .catch(err => setError(err instanceof Error ? err.message : 'Ocorreu um erro desconhecido.'))
            .finally(() => setIsToolRunning(false));
    };
    
    const handleRemoveTool = (toolId: string) => {
        setTools(prev => prev.filter(tool => tool.id !== toolId));
    };

    return (
        <div className="min-h-screen bg-gray-900 text-gray-200 font-sans">
            <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <header className="text-center mb-6">
                    <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-600">
                        Estúdio Criativo IA
                    </h1>
                    <p className="mt-4 text-lg text-gray-400">Dê vida a mundos, personagens e narrativas épicas com o poder da IA.</p>
                </header>
                
                <div className="flex justify-center mb-8">
                    <div className="flex p-1 bg-gray-800 rounded-full">
                        <button onClick={() => setView('studio')} className={`px-6 py-2 text-sm font-semibold rounded-full transition-colors ${view === 'studio' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:bg-gray-700'}`}>Estúdio de Histórias</button>
                        <button onClick={() => setView('tools')} className={`px-6 py-2 text-sm font-semibold rounded-full transition-colors ${view === 'tools' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:bg-gray-700'}`}>Ferramentas de IA</button>
                    </div>
                </div>

                {error && (
                    <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-md relative mb-8" role="alert">
                        <strong className="font-bold">Erro: </strong>
                        <span className="block sm:inline">{error}</span>
                        <button onClick={() => setError(null)} className="absolute top-0 bottom-0 right-0 px-4 py-3" aria-label="Fechar">
                           <span className="text-2xl">&times;</span>
                        </button>
                    </div>
                )}
                
                {view === 'studio' && (activeStoryId === null ? (
                    // --- Library View ---
                    <div className="animate-fade-in">
                         <div className="flex flex-col sm:flex-row justify-between items-center mb-8 gap-4">
                            <h2 className="text-3xl font-bold text-indigo-300">Minha Biblioteca de Histórias</h2>
                            <div className="flex flex-col sm:flex-row items-center gap-4">
                                <SaveStatusIndicator status={saveStatus} />
                                <div className="flex gap-2">
                                    <input type="file" ref={fileInputRef} onChange={handleImport} accept=".json" className="hidden" />
                                    <button onClick={() => fileInputRef.current?.click()} className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md text-white bg-gray-700 hover:bg-gray-600 transition-colors" title="Carregar tudo de um arquivo de backup (.json)">
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" /></svg>
                                        Importar
                                    </button>
                                    <button onClick={handleExport} className="flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md text-white bg-gray-700 hover:bg-gray-600 transition-colors" title="Salvar tudo em um arquivo de backup (.json)">
                                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-4 h-4"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
                                        Exportar Tudo
                                    </button>
                                    <button onClick={handleCreateNewStory} className="px-4 py-2 text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 transition-colors">Nova História +</button>
                                </div>
                            </div>
                        </div>
                        {stories.length > 0 ? (
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                                {stories.map(story => (
                                    <div key={story.id} onClick={() => setActiveStoryId(story.id)} className="cursor-pointer group">
                                        <div className="aspect-[9/16] bg-gray-800 rounded-lg shadow-lg flex items-center justify-center text-center p-2 ring-1 ring-white/10 book-cover-hover">
                                            {story.coverImage ? (
                                                <img src={`data:image/jpeg;base64,${story.coverImage}`} alt={`Capa de ${story.title}`} className="w-full h-full object-cover rounded-md" />
                                            ) : (
                                                <span className="text-gray-400 text-sm">{story.title || "Sem Título"}</span>
                                            )}
                                        </div>
                                        <h3 className="mt-3 text-sm font-semibold truncate group-hover:text-indigo-300 transition-colors">{story.title || "Nova História"}</h3>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-16 px-6 bg-gray-800/50 rounded-2xl ring-1 ring-white/10">
                                <h3 className="text-xl font-semibold text-gray-300">Sua biblioteca está vazia.</h3>
                                <p className="text-gray-400 mt-2">Comece uma nova aventura!</p>
                                <button onClick={handleCreateNewStory} className="mt-6 inline-flex items-center justify-center px-6 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700">Criar Primeira História</button>
                            </div>
                        )}
                    </div>
                ) : activeStory && (
                    // --- Editor View ---
                    <div className="animate-fade-in space-y-8">
                        <button onClick={() => setActiveStoryId(null)} className="mb-4 text-sm text-indigo-300 hover:underline">← Voltar para a Biblioteca</button>
                        
                        {/* Premise */}
                        <div className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10">
                            <label htmlFor="premise" className="block text-xl font-bold mb-3 text-indigo-300">1. Comece com sua ideia</label>
                            <textarea id="premise" rows={3} className="w-full bg-gray-900/70 rounded-lg p-4 focus:ring-2 focus:ring-indigo-500" placeholder="Ex: Um detetive em uma cidade flutuante..." value={activeStory.premise} onChange={(e) => updateActiveStory(() => ({ premise: e.target.value }))} />
                            <div className="mt-4 text-right"><button onClick={handleGenerateConcept} disabled={!activeStory.premise || loading?.type === 'concept'} className="inline-flex items-center justify-center px-6 py-2 text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50 disabled:cursor-not-allowed transition-colors">{loading?.type === 'concept' ? <><Spinner /> Gerando...</> : 'Gerar Conceito'}</button></div>
                        </div>

                        {/* Concept */}
                        {activeStory.title && (
                            <section className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10 animate-fade-in space-y-6">
                                <h2 className="text-3xl font-bold text-indigo-300">{activeStory.title}</h2>
                                <p className="text-sm font-medium text-purple-400 bg-purple-900/50 inline-block px-3 py-1 rounded-full">{activeStory.genre}</p>
                                <p className="text-gray-300 whitespace-pre-wrap leading-relaxed">{activeStory.summary}</p>
                            </section>
                        )}
                        
                        {/* Story Structure Settings */}
                        {activeStory.summary && (
                            <section className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10 animate-fade-in">
                                <h2 className="text-xl font-bold mb-4 text-indigo-300">Estrutura da História</h2>
                                <div className="grid md:grid-cols-2 gap-6">
                                    <div>
                                        <label htmlFor="chapterCount" className="block text-sm font-medium text-gray-300 mb-1">Quantidade de Capítulos</label>
                                        <input type="number" id="chapterCount" className="w-full bg-gray-900/70 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500" value={activeStory.chapterCount} onChange={(e) => updateActiveStory(() => ({ chapterCount: Math.max(1, parseInt(e.target.value, 10) || 1) }))} min="1" />
                                    </div>
                                    <div>
                                        <label htmlFor="wordCount" className="block text-sm font-medium text-gray-300 mb-1">Palavras por Capítulo (aprox.)</label>
                                        <input type="number" id="wordCount" className="w-full bg-gray-900/70 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500" value={activeStory.wordCount} onChange={(e) => updateActiveStory(() => ({ wordCount: Math.max(100, parseInt(e.target.value, 10) || 100) }))} min="100" step="100" />
                                    </div>
                                </div>
                            </section>
                        )}

                        {/* Characters */}
                        {activeStory.summary && (
                            <section className="animate-fade-in">
                                <h2 className="text-3xl font-bold mb-4 text-indigo-300">2. Personagens</h2>
                                <div className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10">
                                    <form onSubmit={handleAddCharacter} className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
                                        <div className="md:col-span-1">
                                            <label htmlFor="charName" className="block text-sm font-medium text-gray-300 mb-1">Nome</label>
                                            <input id="charName" type="text" value={newCharName} onChange={e => setNewCharName(e.target.value)} className="w-full bg-gray-900/70 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500" placeholder="Nome do Personagem" />
                                        </div>
                                        <div className="md:col-span-2">
                                            <label htmlFor="charDesc" className="block text-sm font-medium text-gray-300 mb-1">Descrição</label>
                                            <textarea id="charDesc" value={newCharDesc} onChange={e => setNewCharDesc(e.target.value)} rows={2} className="w-full bg-gray-900/70 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500" placeholder="Descreva aparência, personalidade, etc."></textarea>
                                        </div>
                                        <div className="md:col-span-3 text-right">
                                            <button type="submit" className="inline-flex items-center justify-center px-6 py-2 text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50 transition-colors">Adicionar Personagem</button>
                                        </div>
                                    </form>
                                    
                                    {activeStory.characters.length > 0 && <div className="border-t border-white/10 my-6"></div>}

                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                        {activeStory.characters.map((char, index) => (
                                            <div key={index} className="bg-gray-900/50 p-4 rounded-lg relative">
                                                <h3 className="text-xl font-bold text-purple-300 mb-2">{char.name}</h3>
                                                <p className="text-gray-400 whitespace-pre-wrap text-sm">{char.description}</p>
                                                <button onClick={() => handleRemoveCharacter(index)} className="absolute top-2 right-2 text-gray-500 hover:text-red-400 transition-colors" aria-label={`Remover ${char.name}`}>
                                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM8.28 7.22a.75.75 0 0 0-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 1 0 1.06 1.06L10 11.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L11.06 10l1.72-1.72a.75.75 0 0 0-1.06-1.06L10 8.94 8.28 7.22Z" clipRule="evenodd" /></svg>
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </section>
                        )}

                        {/* Chapters */}
                        {activeStory.characters.length > 0 && (
                            <section className="animate-fade-in">
                                <h2 className="text-3xl font-bold mb-4 text-indigo-300">3. Capítulos</h2>
                                {activeStory.chapters.length > 0 ? (
                                    <div className="space-y-4">{activeStory.chapters.map((chap, index) => (<div key={index} className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10"><h3 className="text-xl font-bold text-purple-300 mb-1">{chap.title}</h3><p className="text-gray-400 italic mb-4">{chap.outline}</p>{chap.content ? (<div className={`whitespace-pre-wrap text-gray-300 p-4 bg-gray-900/50 rounded-lg leading-relaxed ${chap.isNarrating ? 'ring-2 ring-indigo-500' : ''}`}>{chap.content}</div>) : (<button onClick={() => handleGenerateContent(index)} disabled={loading?.type === 'content' && loading.index === index} className="inline-flex items-center justify-center px-4 py-1.5 text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50">{loading?.type === 'content' && loading.index === index ? <><Spinner /> Gerando...</> : 'Escrever Capítulo'}</button>)}<div className="mt-4 flex items-center gap-4 flex-wrap">{chap.content && (<><div className="flex items-center gap-2"><label htmlFor={`style-${index}`} className="text-sm text-gray-400 flex-shrink-0">Estilo de Narração:</label><select id={`style-${index}`} value={chap.narrationStyle} onChange={(e) => handleNarrationStyleChange(index, e.target.value)} className="bg-gray-700 text-white text-sm rounded-md p-1.5 focus:ring-2 focus:ring-indigo-500 border-none">{NARRATION_STYLES.map(style => <option key={style} value={style.toLowerCase()}>{style}</option>)}</select></div>{chap.audio ? (<div className="flex items-center gap-2"><AudioPlayer audioBase64={chap.audio} onPlaybackChange={(isPlaying) => handlePlaybackChange(index, isPlaying)} /><button onClick={() => handleDownloadAudio(index)} className="inline-flex items-center justify-center p-2 rounded-full bg-green-600 hover:bg-green-500 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-green-500" aria-label="Baixar áudio"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-white"><path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" /><path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" /></svg></button></div>) : (<button onClick={() => handleNarrate(index)} disabled={loading?.type === 'audio' && loading.index === index} className="inline-flex items-center justify-center px-4 py-1.5 text-sm font-medium rounded-md text-indigo-200 bg-purple-700 hover:bg-purple-800 disabled:bg-purple-900/50">{loading?.type === 'audio' && loading.index === index ? <><Spinner /> Narrando...</> : 'Ouvir Narração'}</button>)}</>)}</div></div>))}</div>
                                ) : (<div className="text-center bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10"><button onClick={handleGenerateChapters} disabled={loading?.type === 'chapters'} className="inline-flex items-center justify-center px-6 py-2 text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50 transition-colors">{loading?.type === 'chapters' ? <><Spinner /> Gerando...</> : 'Estruturar Capítulos'}</button></div>)}
                            </section>
                        )}

                        {/* Marketing & Publishing */}
                        {activeStory.title && (
                            <section className="animate-fade-in">
                                <h2 className="text-3xl font-bold mb-4 text-indigo-300">4. Marketing e Publicação</h2>
                                <div className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10 space-y-8">
                                    <div>
                                        <label htmlFor="authorName" className="block text-lg font-bold text-purple-300 mb-2">Nome do Autor(a)</label>
                                        <input id="authorName" type="text" className="w-full bg-gray-900/70 rounded-lg p-3 focus:ring-2 focus:ring-indigo-500" placeholder="Digite seu nome ou pseudônimo" value={activeStory.authorName} onChange={(e) => updateActiveStory(() => ({ authorName: e.target.value }))} />
                                    </div>
                                    <div className="grid md:grid-cols-2 gap-6">
                                        <div>
                                            <h3 className="text-lg font-bold text-purple-300 mb-2">Sinopse (Curta)</h3>
                                            {activeStory.synopsis ? <p className="text-gray-400 italic whitespace-pre-wrap text-sm">{activeStory.synopsis}</p> : <button onClick={handleGenerateSynopsis} disabled={loading?.type === 'synopsis'} className="inline-flex items-center px-4 py-1.5 text-sm font-medium rounded-md text-white bg-purple-700 hover:bg-purple-800 disabled:bg-purple-900/50">{loading?.type === 'synopsis' ? <><Spinner/> Gerando...</> : 'Gerar Sinopse'}</button>}
                                        </div>
                                        <div>
                                            <h3 className="text-lg font-bold text-purple-300 mb-2">Descrição (Detalhada)</h3>
                                            {activeStory.description ? <p className="text-gray-400 whitespace-pre-wrap text-sm">{activeStory.description}</p> : <button onClick={handleGenerateDescription} disabled={loading?.type === 'description'} className="inline-flex items-center px-4 py-1.5 text-sm font-medium rounded-md text-white bg-purple-700 hover:bg-purple-800 disabled:bg-purple-900/50">{loading?.type === 'description' ? <><Spinner/> Gerando...</> : 'Gerar Descrição'}</button>}
                                        </div>
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-purple-300 mb-2">Tags de Descoberta</h3>
                                        {activeStory.tags.length > 0 ? (<div className="flex flex-wrap gap-2">{activeStory.tags.map(tag => <span key={tag} className="text-xs bg-gray-700 text-gray-300 px-2.5 py-1 rounded-full">{tag}</span>)}</div>) : <button onClick={handleGenerateTags} disabled={loading?.type === 'tags'} className="inline-flex items-center px-4 py-1.5 text-sm font-medium rounded-md text-white bg-purple-700 hover:bg-purple-800 disabled:bg-purple-900/50">{loading?.type === 'tags' ? <><Spinner/> Gerando...</> : 'Gerar 25 Tags'}</button>}
                                    </div>
                                    <div className="border-t border-white/10 pt-8">
                                        <h3 className="text-xl font-bold text-purple-300 mb-4 text-center">Capa do Livro</h3>
                                        <div className="grid md:grid-cols-2 gap-8 items-start">
                                            <div className="text-center p-4 bg-gray-900/40 rounded-lg">
                                                <h4 className="font-semibold mb-2">Gerador de Capa (IA)</h4>
                                                <p className="text-xs text-gray-400 mb-4">Crie uma capa do zero com base na sua história.</p>
                                                <button onClick={handleGenerateCover} disabled={loading?.type === 'cover'} className="w-full inline-flex items-center justify-center px-4 py-1.5 text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50">{loading?.type === 'cover' ? <><Spinner/> Gerando...</> : 'Gerar com IA'}</button>
                                            </div>
                                            <div className="text-center p-4 bg-gray-900/40 rounded-lg">
                                                <h4 className="font-semibold mb-2">Criador de Capa (Realista)</h4>
                                                <p className="text-xs text-gray-400 mb-4">Envie uma imagem e a IA construirá o cenário ao redor.</p>
                                                <input type="file" id="user-image-upload" className="hidden" accept="image/png, image/jpeg" onChange={handleUserImageUpload}/>
                                                <label htmlFor="user-image-upload" className="w-full cursor-pointer inline-block mb-2 px-4 py-1.5 text-sm font-medium rounded-md text-white bg-gray-600 hover:bg-gray-700">Escolher Imagem</label>
                                                {activeStory.userCoverImage && <img src={`data:${activeStory.userCoverImageMimeType};base64,${activeStory.userCoverImage}`} alt="Pré-visualização" className="w-24 mx-auto rounded-md mb-2"/>}
                                                <button onClick={handleGenerateRealisticCover} disabled={!activeStory.userCoverImage || loading?.type === 'realistic_cover'} className="w-full inline-flex items-center justify-center px-4 py-1.5 text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50 disabled:cursor-not-allowed">{loading?.type === 'realistic_cover' ? <><Spinner/> Criando...</> : 'Criar Capa Realista'}</button>
                                            </div>
                                        </div>
                                        <div className="mt-8">
                                            <h4 className="text-lg font-bold text-center mb-4">Capa Atual</h4>
                                            <div className="max-w-xs mx-auto aspect-[9/16] bg-gray-900/50 rounded-lg flex items-center justify-center ring-1 ring-white/10">
                                                {loading?.type === 'cover' || loading?.type === 'realistic_cover' ? <Spinner /> : (activeStory.coverImage ? <img src={`data:image/jpeg;base64,${activeStory.coverImage}`} alt="Capa do livro" className="w-full h-full object-cover rounded-lg" /> : <p className="text-xs text-gray-500 p-4 text-center">Nenhuma capa gerada. Use uma das opções acima.</p>)}
                                            </div>
                                            <div className="text-center mt-4">
                                                <button onClick={handleDownloadCover} disabled={!activeStory.coverImage} className="inline-flex items-center justify-center px-6 py-2 text-base font-medium rounded-md text-white bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed">
                                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 mr-2"><path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" /><path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" /></svg>
                                                    Baixar Capa
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </section>
                        )}
                    </div>
                ))}

                {view === 'tools' && (
                     <div className="animate-fade-in space-y-8">
                        <div className="text-center">
                            <h2 className="text-3xl font-bold text-indigo-300">Ferramenta de IA Mágica</h2>
                            <p className="text-gray-400 mt-2">Crie sua própria ferramenta de IA com um único prompt.</p>
                        </div>
                        <div className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10">
                            <label htmlFor="tool-prompt" className="block text-xl font-bold mb-3 text-indigo-300">1. Defina sua Ferramenta</label>
                            <textarea id="tool-prompt" rows={4} className="w-full bg-gray-900/70 rounded-lg p-4 focus:ring-2 focus:ring-indigo-500" placeholder="Ex: Você é um mestre de haikus. Transforme qualquer texto que eu fornecer em um haiku no formato 5-7-5." value={newToolPrompt} onChange={(e) => setNewToolPrompt(e.target.value)} />
                            <div className="mt-4 text-right">
                                <button onClick={handleCreateTool} disabled={!newToolPrompt || loading?.type === 'tool_analysis'} className="inline-flex items-center justify-center px-6 py-2 text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50 disabled:cursor-not-allowed transition-colors">
                                    {loading?.type === 'tool_analysis' ? <><Spinner /> Criando...</> : 'Criar Ferramenta Mágica'}
                                </button>
                            </div>
                        </div>

                        <div>
                           <h2 className="text-2xl font-bold mb-4 text-indigo-300">2. Sua Biblioteca de Ferramentas</h2>
                           {tools.length > 0 ? (
                               <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                   {tools.map(tool => (
                                       <div key={tool.id} className="bg-gray-800/50 p-6 rounded-2xl ring-1 ring-white/10 flex flex-col justify-between">
                                            <div>
                                               <div className="flex justify-between items-start">
                                                   <h3 className="text-xl font-bold text-purple-300 mb-2"><span className="text-2xl mr-2">{tool.icon}</span>{tool.name}</h3>
                                                   <button onClick={() => handleRemoveTool(tool.id)} className="text-gray-500 hover:text-red-400 transition-colors" aria-label={`Remover ${tool.name}`}>
                                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM8.28 7.22a.75.75 0 0 0-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 1 0 1.06 1.06L10 11.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L11.06 10l1.72-1.72a.75.75 0 0 0-1.06-1.06L10 8.94 8.28 7.22Z" clipRule="evenodd" /></svg>
                                                    </button>
                                               </div>
                                               <p className="text-gray-400 text-sm mb-4">{tool.description}</p>
                                           </div>
                                            <button onClick={() => { setActiveTool(tool); setToolInput(''); setToolOutput(''); }} className="w-full mt-2 inline-flex items-center justify-center px-4 py-2 text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 transition-colors">
                                               Usar Ferramenta
                                           </button>
                                       </div>
                                   ))}
                               </div>
                           ) : (
                               <div className="text-center py-12 px-6 bg-gray-800/50 rounded-2xl ring-1 ring-white/10">
                                   <h3 className="text-xl font-semibold text-gray-300">Nenhuma ferramenta criada ainda.</h3>
                                   <p className="text-gray-400 mt-2">Use o formulário acima para criar sua primeira ferramenta mágica!</p>
                               </div>
                           )}
                        </div>
                    </div>
                )}
            </main>
            
            {/* --- Chat Assistant --- */}
            <button onClick={openChat} className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-500 to-indigo-600 text-white p-4 rounded-full shadow-lg hover:scale-110 transition-transform focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-indigo-500" aria-label="Abrir Assistente Criativo">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6"><path d="M9.161 2.217a3.837 3.837 0 0 1 5.678 0 3.837 3.837 0 0 1 0 5.678l-6.53 6.53a.75.75 0 0 1-1.06-1.061l6.53-6.53a2.338 2.338 0 0 0-3.307-3.307l-6.53 6.53a.75.75 0 0 1-1.06-1.06l6.53-6.53Z" /><path d="m12.44 9.47.669-.67a3.837 3.837 0 0 1 5.678 0 3.837 3.837 0 0 1 0 5.678l-6.53 6.53a.75.75 0 0 1-1.06-1.06l6.53-6.53a2.338 2.338 0 0 0-3.306-3.307l-6.53 6.53a.75.75 0 0 1-1.06-1.06l6.53-6.53-.669.67a.75.75 0 0 1-1.06-1.06l.669-.67-.669.67a.75.75 0 0 1-1.06-1.06l.669-.67-.669.67a.75.75 0 1 1-1.06-1.06l.669-.67a3.837 3.837 0 0 1 5.678 0 3.837 3.837 0 0 1 0 5.678l-6.53 6.53a.75.75 0 0 1-1.06-1.06l6.53-6.53a2.338 2.338 0 0 0-3.307-3.307l-6.53 6.53a.75.75 0 1 1-1.06-1.06l6.53-6.53-.669.67a.75.75 0 0 1-1.06-1.06l.669-.67Z" /></svg>
            </button>
            
            {isChatOpen && (
                <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-40 animate-fade-in" onClick={() => setIsChatOpen(false)}></div>
            )}
            {isChatOpen && (
                <div className="fixed inset-0 sm:inset-auto sm:bottom-6 sm:right-6 sm:max-w-lg w-full h-full sm:h-[70vh] flex flex-col bg-gray-800/80 backdrop-blur-xl border border-white/10 shadow-2xl rounded-2xl z-50 animate-fade-in">
                    <header className="flex items-center justify-between p-4 border-b border-white/10">
                         <h2 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-600">Assistente Criativo Kekakekinhafan</h2>
                        <button onClick={() => setIsChatOpen(false)} className="text-gray-400 hover:text-white">&times;</button>
                    </header>
                    <div ref={chatBodyRef} className="flex-1 p-4 overflow-y-auto space-y-4">
                        {chatHistory.map((msg) => (
                            <div key={msg.id} className={`flex items-end gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                {msg.role === 'model' && <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex-shrink-0"></div>}
                                <div className={`max-w-sm md:max-w-md p-3 rounded-2xl ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-br-none' : 'bg-gray-700 text-gray-200 rounded-bl-none'}`}>
                                    {typeof msg.content === 'string' ? <p className="whitespace-pre-wrap">{msg.content}{isChatLoading && chatHistory[chatHistory.length - 1].id === msg.id && <span className="inline-block w-1 h-4 bg-white animate-ping ml-1"></span>}</p> : <img src={msg.content.imageUrl} alt={msg.content.alt} className="rounded-lg" />}
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="p-4 border-t border-white/10">
                        <div className="mb-2 flex gap-2 overflow-x-auto pb-2">
                             {["Sugira um plot twist", "Descreva uma cidade fantástica", "Crie um diálogo tenso"].map(prompt => (
                                <button key={prompt} onClick={() => handleSendChatMessage(prompt)} className="text-xs px-3 py-1 rounded-full bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors whitespace-nowrap">{prompt}</button>
                            ))}
                        </div>
                        <form onSubmit={(e) => { e.preventDefault(); handleSendChatMessage(chatInput); }} className="flex items-center gap-2">
                            <input type="text" value={chatInput} onChange={e => setChatInput(e.target.value)} placeholder="Pergunte ou use /imagem [prompt]..." className="flex-1 bg-gray-900/70 rounded-lg p-2 focus:ring-2 focus:ring-indigo-500 text-sm" />
                             <button type="button" disabled title="Conversar com arquivos (em breve)" className="p-2 text-gray-500 cursor-not-allowed"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M18.97 3.659a2.25 2.25 0 0 0-3.182 0l-10.94 10.94a2.25 2.25 0 1 0 3.182 3.182L18.97 6.841a.75.75 0 0 0-1.06-1.06l-10.94 10.94a.75.75 0 1 1-1.06-1.061l10.94-10.94a2.25 2.25 0 0 0 0-3.182Z" clipRule="evenodd" /></svg></button>
                            <button type="submit" disabled={isChatLoading} className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-900 text-white transition-colors">{ isChatLoading ? <Spinner /> : <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path d="M3.478 2.404a.75.75 0 0 0-.926.941l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.404Z" /></svg>}</button>
                        </form>
                    </div>
                </div>
            )}
            {activeTool && (
                <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in" onClick={() => setActiveTool(null)}>
                    <div className="w-full max-w-2xl bg-gray-800/80 backdrop-blur-xl border border-white/10 shadow-2xl rounded-2xl z-50" onClick={(e) => e.stopPropagation()}>
                        <header className="flex items-center justify-between p-4 border-b border-white/10">
                            <h2 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-indigo-600 flex items-center gap-3"><span className="text-2xl">{activeTool.icon}</span>{activeTool.name}</h2>
                            <button onClick={() => setActiveTool(null)} className="text-gray-400 hover:text-white">&times;</button>
                        </header>
                        <div className="p-6 space-y-4">
                            <div>
                                <label htmlFor="tool-input" className="block text-sm font-medium text-gray-300 mb-1">Entrada</label>
                                <textarea id="tool-input" rows={5} value={toolInput} onChange={(e) => setToolInput(e.target.value)} className="w-full bg-gray-900/70 rounded-lg p-3 focus:ring-2 focus:ring-indigo-500" placeholder="Insira seu texto aqui..." />
                            </div>
                             <div className="text-center">
                                <button onClick={handleExecuteTool} disabled={isToolRunning || !toolInput} className="inline-flex items-center justify-center px-6 py-2 text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-800/50 transition-colors">
                                    {isToolRunning ? <><Spinner /> Executando...</> : 'Executar'}
                                </button>
                            </div>
                            {isToolRunning && !toolOutput && <div className="text-center text-gray-400">Gerando resposta...</div>}
                            {toolOutput && (
                                <div>
                                    <label className="block text-sm font-medium text-gray-300 mb-1">Saída</label>
                                    <div className="w-full bg-gray-900/50 rounded-lg p-4 whitespace-pre-wrap text-gray-300 leading-relaxed max-h-64 overflow-y-auto">{toolOutput}</div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}