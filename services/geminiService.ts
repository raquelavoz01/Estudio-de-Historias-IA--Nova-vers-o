import { GoogleGenAI, Type, Modality } from "@google/genai";
import type { Story, Character, Chapter } from '../types';

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
4.  **Foco:** O foco principal é o capítulo '${chapter.title}', cujo esboço é: '${chapter.outline}'.

**Contexto da História Fornecido:**
${storyContext}

Agora, escreva o conteúdo completo e detalhado para o capítulo '${chapter.title}' em português.`,
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
            prompt: `Crie uma capa de livro para a plataforma Wattpad, formato 9:16.
**Título:** '${title}'
**Autor:** '${authorName}'
**Enredo:** '${summary}'

**INSTRUÇÃO CRÍTICA DE ORTOGRAFIA - LEIA COM ATENÇÃO:**
1.  **TOLERÂNCIA ZERO PARA ERROS:** A ortografia de TODO o texto deve ser 100% correta, seguindo o português padrão. Esta é a regra mais importante.
2.  **EXEMPLO OBRIGATÓRIO - 'CORAÇÃO':** A palavra "coração" DEVE ser escrita com 'ç' e 'ão'. NÃO escreva 'coraran', 'coraan', ou qualquer outra variação incorreta. Verifique a grafia antes de gerar a imagem.
3.  **VERIFICAÇÃO DUPLA:** Antes de finalizar a imagem, verifique se cada letra em '${title}' e '${authorName}' está perfeitamente correta.

**Outras Instruções:**
- **Texto Contido:** O título e o autor DEVEM estar completamente dentro das margens da capa.
- **Legibilidade:** O texto deve ser claro, legível e bem posicionado.
- **Estilo:** Arte vibrante e dramática para atrair um público jovem.
- **Idioma:** Todo o texto na capa DEVE estar em português.`,
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
                        text: `Sua tarefa é criar uma capa de livro profissional no formato 9:16, usando a imagem realista fornecida.

**Título:** '${title}'
**Autor:** '${authorName}'

**INSTRUÇÃO CRÍTICA DE ORTOGRAFIA - LEIA COM ATENÇÃO:**
1.  **TOLERÂNCIA ZERO PARA ERROS:** Sua tarefa mais importante é garantir a ortografia 100% correta de todo o texto, em português padrão.
2.  **EXEMPLO OBRIGATÓRIO - 'CORAÇÃO':** Se a palavra "coração" aparecer, ela DEVE ser escrita com 'ç' e 'ão'. É expressamente proibido usar variações como 'coraran' ou 'coraan'. A falha em escrever "coração" corretamente resultará em uma capa inaceitável.
3.  **VERIFICAÇÃO DUPLA:** Revise a grafia do título '${title}' e do autor '${authorName}' para garantir que não haja absolutamente nenhum erro.

**Outras Instruções:**
- **Integração de Imagem:** Use a imagem fornecida como o elemento central. Construa um cenário de fundo dramático e atmosférico ao redor dela.
- **Posicionamento do Texto:** O título e o autor DEVEM ser posicionados de forma legível e COMPLETAMENTE DENTRO das margens da capa.
- **Idioma:** Todo o texto na capa DEVE estar em português.`,
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