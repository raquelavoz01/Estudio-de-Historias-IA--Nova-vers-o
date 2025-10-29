import { GoogleGenAI, Type, Modality } from "@google/genai";
import type { Story, Character, Chapter, MagicTool } from '../types';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });

export const generateConcept = async (premise: string): Promise<Pick<Story, 'title' | 'genre' | 'summary'>> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: `Você é um assistente de escrita criativa. Com base na premissa a seguir, crie um conceito de história detalhado. A premissa é: '${premise}'. Gere um título cativante, um gênero, e um resumo expandido da trama em português.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            genre: { type: Type.STRING },
            summary: { type: Type.STRING }
          },
          required: ["title", "genre", "summary"]
        },
      },
    });
    
    return JSON.parse(response.text);
  } catch (error) {
    console.error("Error generating concept:", error);
    throw new Error("Não foi possível gerar o conceito da história.");
  }
};

export const generateChapterOutlines = async (summary: string, characters: Character[], chapterCount: number): Promise<Pick<Chapter, 'title' | 'outline'>[]> => {
  const characterProfiles = characters.map(c => `- ${c.name}: ${c.description}`).join('\n');
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: `Você é um planejador de histórias mestre. Sua tarefa é criar um esboço de capítulos coeso e envolvente.

**Instruções Cruciais:**
1.  **Respeite o Contexto:** Baseie o esboço estritamente no resumo da história e nas descrições detalhadas dos personagens fornecidas abaixo. A personalidade e o histórico de cada personagem DEVEM influenciar os eventos dos capítulos.
2.  **Estrutura:** Crie um esboço para um livro com ${chapterCount} capítulos.
3.  **Formato:** Forneça um título e um resumo de uma frase para cada capítulo.

**Resumo da História:**
'${summary}'

**Personagens Detalhados:**
${characterProfiles}

Agora, gere os esboços dos capítulos em português.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              title: { type: Type.STRING },
              outline: { type: Type.STRING }
            },
            required: ["title", "outline"]
          }
        },
      },
    });
    return JSON.parse(response.text);
  } catch (error) {
    console.error("Error generating chapter outlines:", error);
    throw new Error("Não foi possível gerar os capítulos.");
  }
};


export const generateChapterContent = async (summary: string, chapter: Pick<Chapter, 'title' | 'outline'>, storyContext: string, wordCount: number): Promise<string> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: `Você é um romancista mestre. Sua tarefa é escrever um capítulo de um livro, respeitando fielmente o contexto fornecido.

**Instruções Cruciais:**
1.  **Continue a História:** Escreva o conteúdo para o capítulo '${chapter.title}'. A história deve progredir a partir do contexto fornecido. NÃO repita ou reescreva versões de capítulos anteriores. Crie conteúdo NOVO que faça a trama avançar.
2.  **Fidelidade aos Personagens:** Utilize as descrições detalhadas dos personagens fornecidas no contexto. As ações, diálogos e pensamentos dos personagens DEVEM ser consistentes com suas personalidades e históricos definidos. NÃO os faça agir fora do personagem.
3.  **Extensão Obrigatória:** O capítulo DEVE ter PELO MENOS ${wordCount} palavras. Seja detalhado, descritivo e desenvolva a trama e os personagens profundamente.
4.  **Estilo de Escrita Claro e Direto:** A prioridade é a clareza. Escreva de forma que a história seja fácil de entender. Evite linguagem excessivamente poética, metáforas obscuras e frases complexas que possam confundir o leitor. A prosa deve ser envolvente, mas sempre acessível e direta.
5.  **Formatação para Publicação:** O texto DEVE ser formatado para fácil leitura em plataformas online (como Wattpad).
    -   **Parágrafos Curtos:** Separe ações, descrições e pensamentos em parágrafos distintos e curtos. Evite blocos de texto longos e densos.
    -   **Diálogos Corretos:** Cada fala de um personagem DEVE estar em seu próprio parágrafo. Inicie cada diálogo com um travessão (—). A formatação deve ser impecável para que possa ser copiada e colada diretamente.
    -   **Exemplo de Diálogo:**
        — Eu não posso acreditar que você fez isso.
        — Foi necessário — ele respondeu, sem desviar o olhar.
6.  **Foco:** O foco principal é o capítulo '${chapter.title}', cujo esboço é: '${chapter.outline}'.

**Contexto da História Fornecido:**
${storyContext}

Agora, escreva o conteúdo completo e detalhado para o capítulo '${chapter.title}' em português, seguindo rigorosamente todas as instruções, especialmente as de formatação.`,
    });
    return response.text;
  } catch (error) {
    console.error("Error generating chapter content:", error);
    throw new Error("Não foi possível gerar o conteúdo do capítulo.");
  }
};

export const narrateText = async (text: string, style: string): Promise<string> => {
    if (!text) throw new Error("O texto para narração não pode estar vazio.");
    try {
        const prompt = `Leia o seguinte texto em um tom de contador de histórias calmo, envolvente e marcadamente ${style}, em português: ${text}`;
        
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash-preview-tts",
            contents: [{ parts: [{ text: prompt }] }],
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: {
                        prebuiltVoiceConfig: { voiceName: 'Puck' },
                    },
                },
            },
        });
        const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
        if (!base64Audio) {
            throw new Error("Nenhum dado de áudio recebido.");
        }
        return base64Audio;
    } catch (error) {
        console.error("Error narrating text:", error);
        throw new Error("Não foi possível gerar a narração de áudio.");
    }
};

export const generateSynopsis = async (title: string, summary: string): Promise<string> => {
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-pro',
            contents: `Você é um editor de livros experiente. Crie uma sinopse curta e impactante (cerca de 100 palavras) para um livro com o título '${title}' e o seguinte resumo: '${summary}'. A sinopse deve ser cativante e adequada para a contracapa de um livro ou uma plataforma online como o Wattpad.`,
        });
        return response.text;
    } catch (error) {
        console.error("Error generating synopsis:", error);
        throw new Error("Não foi possível gerar a sinopse.");
    }
};

export const generateDescription = async (title: string, summary: string): Promise<string> => {
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-pro',
            contents: `Crie uma descrição detalhada para um livro com o título '${title}' e o seguinte resumo: '${summary}'. A descrição deve ter entre 150-200 palavras, expandindo a trama, os conflitos principais e os temas, para ser usada em uma página de produto de uma loja online.`,
        });
        return response.text;
    } catch (error) {
        console.error("Error generating description:", error);
        throw new Error("Não foi possível gerar a descrição.");
    }
};

export const generateTags = async (title: string, summary: string, genre: string): Promise<string[]> => {
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-pro',
            contents: `Para um livro de gênero '${genre}' com o título '${title}' e resumo '${summary}', gere exatamente 25 tags de palavras-chave individuais para ajudar na descoberta em plataformas online. As tags devem ser relevantes, específicas e variadas.`,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        tags: {
                            type: Type.ARRAY,
                            items: { type: Type.STRING },
                            description: "Uma lista de exatamente 25 tags de palavra-chave."
                        }
                    },
                    required: ["tags"]
                }
            }
        });
        const result = JSON.parse(response.text);
        return result.tags;
    } catch (error) {
        console.error("Error generating tags:", error);
        throw new Error("Não foi possível gerar as tags.");
    }
};


export const generateCoverImage = async (title: string, summary: string, authorName: string): Promise<string> => {
    try {
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: `**REGRAS ABSOLUTAS PARA GERAÇÃO DE IMAGEM:**

**REGRA 1: PROIBIDO ESCREVER QUALQUER TEXTO.**
- A imagem final NÃO DEVE CONTER NENHUMA LETRA, PALAVRA OU TEXTO. NENHUM MESMO.
- A tarefa é 100% visual.

**REGRA 2: CRIAR ESPAÇOS VAZIOS PARA TEXTO.**
- Crie um espaço VAZIO e proeminente na parte superior para o TÍTULO.
- Crie um segundo espaço VAZIO, menor, na parte inferior para o NOME DO AUTOR.
- Estes espaços devem ser parte do design, mas completamente limpos, sem texto.

**REGRA 3: INSPIRAÇÃO VISUAL.**
- A arte deve ser inspirada pelo seguinte enredo. Use-o para criar a cena, o clima e os elementos visuais.
- Enredo para inspiração: '${summary}'

**REGRA 4: FORMATO E ESTILO.**
- Formato: 9:16 (vertical).
- Estilo: Vibrante, dramático, para um público jovem (estilo Wattpad).

**AVISO FINAL: Se você escrever QUALQUER texto na imagem, a tarefa será considerada uma falha completa. O resultado deve ser uma arte de capa sem nenhuma palavra.**`,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: '9:16',
            },
        });

        const base64ImageBytes = response.generatedImages[0].image.imageBytes;
        if (!base64ImageBytes) {
            throw new Error("Nenhuma imagem foi gerada.");
        }
        return base64ImageBytes;
    } catch (error) {
        console.error("Error generating cover image:", error);
        throw new Error("Não foi possível gerar a capa do livro.");
    }
};

export const generateRealisticCover = async (title: string, summary: string, authorName: string, userImageBase64: string, mimeType: string): Promise<string> => {
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: {
                parts: [
                    {
                        inlineData: {
                            data: userImageBase64,
                            mimeType: mimeType,
                        },
                    },
                    {
                        text: `**REGRAS ABSOLUTAS PARA EDIÇÃO DE IMAGEM:**

**REGRA 1: PROIBIDO ADICIONAR QUALQUER TEXTO.**
- A imagem final NÃO DEVE CONTER NENHUMA LETRA, PALAVRA OU TEXTO NOVO.
- A tarefa é 100% visual, focada em integrar a imagem fornecida em um cenário de capa de livro.

**REGRA 2: CRIAR ESPAÇOS VAZIOS PARA TEXTO.**
- Adicione elementos de design (fundos, efeitos de luz, etc.) ao redor da imagem fornecida.
- Dentro desse novo design, crie um espaço VAZIO e estilizado para o TÍTULO.
- Crie também um segundo espaço VAZIO para o NOME DO AUTOR.
- **NÃO ESCREVA NADA NESTES ESPAÇOS.**

**REGRA 3: INSPIRAÇÃO VISUAL.**
- O design geral deve ser inspirado pelo resumo da história. Use-o para definir o clima e o estilo.
- Resumo para inspiração: '${summary}'

**REGRA 4: INTEGRAÇÃO DA IMAGEM**
- Use a imagem fornecida como o elemento principal.
- Transforme-a em uma capa de livro profissional e atmosférica adicionando um fundo e outros elementos visuais.

**AVISO FINAL: Se você escrever QUALQUER texto na imagem, a tarefa será considerada uma falha completa. O resultado deve ser a imagem do usuário integrada em uma arte de capa, sem nenhuma palavra adicionada.**`,
                    },
                ],
            },
            config: {
                responseModalities: [Modality.IMAGE],
            },
        });
        for (const part of response.candidates[0].content.parts) {
            if (part.inlineData) {
                return part.inlineData.data;
            }
        }
        throw new Error("Nenhuma imagem foi gerada na resposta.");
    } catch (error) {
        console.error("Error generating realistic cover:", error);
        throw new Error("Não foi possível gerar a capa realista.");
    }
};

export const generateConceptArt = async (prompt: string): Promise<string> => {
    try {
        const response = await ai.models.generateImages({
            model: 'imagen-4.0-generate-001',
            prompt: `Arte conceitual para uma história. Uma ilustração vívida e imaginativa de: '${prompt}'. Iluminação cinematográfica, detalhada, estilo de fantasia épica.`,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: '16:9',
            },
        });

        const base64ImageBytes = response.generatedImages[0].image.imageBytes;
        if (!base64ImageBytes) {
            throw new Error("Nenhuma imagem foi gerada.");
        }
        return base64ImageBytes;
    } catch (error) {
        console.error("Error generating concept art:", error);
        throw new Error("Não foi possível gerar a arte conceitual.");
    }
};

export const analyzeToolPrompt = async (prompt: string): Promise<Pick<MagicTool, 'name' | 'description' | 'icon'>> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: `Analise o seguinte prompt de IA, que define uma ferramenta. Crie um nome curto e cativante para a ferramenta (em português), uma descrição de uma frase do que ela faz (em português) e sugira um único emoji para representá-la.

Prompt da Ferramenta: "${prompt}"

Responda apenas com o JSON.`,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            name: { type: Type.STRING, description: "Nome curto para a ferramenta (2-3 palavras)" },
            description: { type: Type.STRING, description: "Descrição de uma frase." },
            icon: { type: Type.STRING, description: "Um único emoji." }
          },
          required: ["name", "description", "icon"]
        },
      },
    });
    
    return JSON.parse(response.text);
  } catch (error) {
    console.error("Error analyzing tool prompt:", error);
    throw new Error("Não foi possível analisar o prompt da ferramenta.");
  }
};

export const executeMagicTool = async (toolPrompt: string, userInput: string): Promise<string> => {
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: userInput,
      config: {
        systemInstruction: toolPrompt,
      },
    });
    return response.text;
  } catch (error) {
    console.error("Error executing magic tool:", error);
    throw new Error("Não foi possível executar a ferramenta de IA.");
  }
};