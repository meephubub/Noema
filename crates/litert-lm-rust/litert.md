[**`Conversation`**](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) is a high-level API, representing a
single, stateful conversation with the LLM and is the recommended entry point
for most users. It internally manages a [`Session`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h) and handles
complex data processing tasks. These tasks include maintaining the initial
context, managing tool definitions, preprocessing multimodal data, and applying
Jinja prompt templates with role-based message formatting.

## Conversation API Workflow

The typical lifecycle for using the Conversation API is:

1. **Create an [`Engine`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h)** : Initialize a single [`Engine`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h) with the model path and configuration. This is a heavyweight object that holds the model weights.
2. **Create a [`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h)** : Use the [`Engine`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h) to create one or more lightweight [`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) objects.
3. **Send Message** : Use the [`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) object's methods to send messages to the LLM and receive responses, effectively enabling a chat-like interaction.

Below is the simplest way to send message and get model response. It is
recommended for most use cases. It mirrors
[Gemini Chat APIs](https://ai.google.dev/gemini-api/docs/text-generation#multi-turn-conversations).

- [`SendMessage`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h): A blocking call that takes user input and returns the complete model response.
- [`SendMessageAsync`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h): A non-blocking call that streams the model's response back token-by-token through callbacks.

Here are example code snippet:

### Text only content

    #include "runtime/engine/engine.h"

    // ...

    // 1. Define model assets and engine settings.
    auto model_assets = ModelAssets::Create(model_path);
    CHECK_OK(model_assets);

    auto engine_settings = EngineSettings::CreateDefault(
        model_assets,
        /*backend=*/litert::lm::Backend::CPU);

    // 2. Create the main Engine object.
    absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(engine_settings);
    CHECK_OK(engine);

    // 3. Create a Conversation
    auto conversation_config = ConversationConfig::CreateDefault(**engine);
    CHECK_OK(conversation_config)
    absl::StatusOr<std::unique_ptr<Conversation>> conversation = Conversation::Create(**engine, *conversation_config);
    CHECK_OK(conversation);

    // 4. Send message to the LLM with blocking call.
    absl::StatusOr<Message> model_message = (*conversation)->SendMessage(
        JsonMessage{
            {"role", "user"},
            {"content", "What is the tallest building in the world?"}
        });
    CHECK_OK(model_message);

    // 5. Print the model message.
    std::cout << *model_message << std::endl;

    // 6. Send message to the LLM with asynchronous call
    // where CreatePrintMessageCallback is a users implemented callback that would
    // process the message once a chunk of message output is received.
    std::stringstream captured_output;
    (*conversation)->SendMessageAsync(
        JsonMessage{
            {"role", "user"},
            {"content", "What is the tallest building in the world?"}
        },
        CreatePrintMessageCallback(std::stringstream& captured_output)
    );
    // Wait until asynchronous finish or timeout.
    *engine->WaitUntilDone(absl::Seconds(10));

<br />

<br />

Example `CreatePrintMessageCallback`

<br />

<br />

    absl::AnyInvocable<void(absl::StatusOr<Message>)> CreatePrintMessageCallback(
        std::stringstream& captured_output) {
      return [&captured_output](absl::StatusOr<Message> message) {
        if (!message.ok()) {
          std::cout << message.status().message() << std::endl;
          return;
        }
        if (auto json_message = std::get_if<JsonMessage>(&(*message))) {
          if (json_message->is_null()) {
            std::cout << std::endl << std::flush;
            return;
          }
          ABSL_CHECK_OK(PrintJsonMessage(*json_message, captured_output,
                                         /*streaming=*/true));
        }
      };
    }

    absl::Status PrintJsonMessage(const JsonMessage& message,
                                  std::stringstream& captured_output,
                                  bool streaming = false) {
      if (message["content"].is_array()) {
        for (const auto& content : message["content"]) {
          if (content["type"] == "text") {
            captured_output << content["text"].get<std::string>();
            std::cout << content["text"].get<std::string>();
          }
        }
        if (!streaming) {
          captured_output << std::endl << std::flush;
          std::cout << std::endl << std::flush;
        } else {
          captured_output << std::flush;
          std::cout << std::flush;
        }
      } else if (message["content"]["text"].is_string()) {
        if (!streaming) {
          captured_output << message["content"]["text"].get<std::string>()
                          << std::endl
                          << std::flush;
          std::cout << message["content"]["text"].get<std::string>() << std::endl
                    << std::flush;
        } else {
          captured_output << message["content"]["text"].get<std::string>()
                          << std::flush;
          std::cout << message["content"]["text"].get<std::string>() << std::flush;
        }
      } else {
        return absl::InvalidArgumentError("Invalid message: " + message.dump());
      }
      return absl::OkStatus();
    }

<br />

<br />

### 🔴 New: Multi-Token Prediction (MTP)

Multi-Token Prediction (MTP) is a performance optimization that significantly
accelerates decode speeds. MTP is universally recommended for all tasks on GPU
backends.

To use MTP, you need to enable speculative decoding in the advanced settings of
the engine configuration.

    // 1. Define model assets and engine settings.
    auto model_assets = ModelAssets::Create(model_path);
    CHECK_OK(model_assets);

    auto engine_settings = EngineSettings::CreateDefault(
        model_assets,
        /*backend=*/litert::lm::Backend::GPU);
    CHECK_OK(engine_settings);

    // 2. Enable MTP via speculative decoding in advanced settings.
    litert::lm::AdvancedSettings advanced_settings;
    advanced_settings.enable_speculative_decoding = true;
    engine_settings->GetMutableMainExecutorSettings().SetAdvancedSettings(
        advanced_settings);

    // 3. Create the main Engine object.
    absl::StatusOr<std::unique_ptr<Engine>> engine = Engine::CreateEngine(
        *engine_settings);
    CHECK_OK(engine);

    // The same steps to create Conversation and send messages as above...

### Multimodal data content

    // To use multimodality, the engine must be created with vision and audio
    // backend depending on the multimodality to be used
    auto engine_settings = EngineSettings::CreateDefault(
        model_assets,
        /*backend=*/litert::lm::Backend::CPU,
        /*vision_backend*/litert::lm::Backend::GPU,
        /*audio_backend*/litert::lm::Backend::CPU,
    );

    // The same steps to create Engine and Conversation as above...

    // Send message to the LLM with image data.
    absl::StatusOr<Message> model_message = (*conversation)->SendMessage(
        JsonMessage{
            {"role", "user"},
            {"content", { // Now content must be an array.
              {
                {"type", "text"}, {"text", "Describe the following image: "}
              },
              {
                {"type", "image"}, {"path", "/file/path/to/image.jpg"}
              }
            }},
        });
    CHECK_OK(model_message);

    // Print the model message.
    std::cout << *model_message << std::endl;

    // Send message to the LLM with audio data.
    model_message = (*conversation)->SendMessage(
        JsonMessage{
            {"role", "user"},
            {"content", { // Now content must be an array.
              {
                {"type", "text"}, {"text", "Transcribe the audio: "}
              },
              {
                {"type", "audio"}, {"path", "/file/path/to/audio.wav"}
              }
            }},
        });
    CHECK_OK(model_message);

    // Print the model message.
    std::cout << *model_message << std::endl;

    // The content can include multiple image or audio data.
    model_message = (*conversation)->SendMessage(
        JsonMessage{
            {"role", "user"},
            {"content", { // Now content must be an array.
              {
                {"type", "text"}, {"text", "First briefly describe the two images "}
              },
              {
                {"type", "image"}, {"path", "/file/path/to/image1.jpg"}
              },
              {
                {"type", "text"}, {"text", "and "}
              },
              {
                {"type", "image"}, {"path", "/file/path/to/image2.jpg"}
              },
              {
                {"type", "text"}, {"text", " then transcribe the content in the audio"}
              },
              {
                {"type", "audio"}, {"path", "/file/path/to/audio.wav"}
              }
            }},
        });
    CHECK_OK(model_message);

    // Print the model message.
    std::cout << *model_message << std::endl;

### Use Conversation with Tools

Refer to [Advanced Usage](https://developers.google.com/edge/litert-lm/cpp#advanced-usage) for detailed Tool Usage with
Conversation API

## Components in Conversation

[`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) could be regarded as a delegate for users to
maintain [`Session`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h) and complicated data processing before sending the
data to Session.

### I/O Types

The core input and output format for the Conversation API is
[`Message`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h). Currently, this is implemented as
[`JsonMessage`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h), which is a type alias for
[`ordered_json`](https://json.nlohmann.me/api/ordered_json/), a flexible nested key-value data structure.

The [`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) API operates on a message-in-message-out
basis, mimicking a typical chat experience. The flexibility of
[`Message`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h) allows users to include arbitrary fields as needed by
specific prompt templates or LLM models, enabling LiteRT-LM to support a wide
variety of models.

While there isn't a single rigid standard, most prompt templates and models
expect [`Message`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h) to follow conventions similar to those used in the
[Gemini API Content](https://ai.google.dev/api/caching#Content) or the
[OpenAI Message structure](https://platform.openai.com/docs/api-reference/chat/create).

[`Message`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h) must contain `role`, representing who the message is sent
from. `content` can be as simple as a text string.

    {
      "role": "model", // Represent who the message is sent from.
      "content": "Hello World!" // Naive text only content.
    }

For multimodal data input, `content` is a list of `part`. Again `part` is not a
predefined data structure but a
[ordered key-value pair data type](https://json.nlohmann.me/api/ordered_json/). The specific fields depend on
what the prompt template and the model expect.

    {
      "role": "user",
      "content": [  // Multimodal content.
        // Now the content is composed of parts
        {
          "type": "text",
          "text": "Describe the image in details: "
        },
        {
          "type": "image",
          "path": "/path/to/image.jpg"
        }
      ]
    }

For multimodal `part`, we support the following format handled by
[`data_utils.h`](https://github.com/google-ai-edge/LiteRT-LM/blob/e5ead3aa409b8e631111d487aaa5e54e91ead747/runtime/conversation/model_data_processor/data_utils.h#L26-L57)

    {
      "type": "text",
      "text": "this is a text"
    }

    {
      "type": "image",
      "path": "/path/to/image.jpg"
    }

    {
      "type": "image",
      "blob": "base64 encoded image bytes as string",
    }

    {
      "type": "audio",
      "path": "/path/to/audio.wav"
    }

    {
      "type": "audio",
      "blob": "base64 encoded audio bytes as string",
    }

### Prompt Template

To maintain flexibility for variant models, [PromptTemplate](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/prompt_template.h) is
implemented as a thin wrapper around [Minja](https://github.com/google/minja).
Minja is a C++ implementation of the [Jinja template engine](https://jinja.palletsprojects.com/en/stable/), which
processes JSON input to generate formatted prompts.

The [Jinja template engine](https://jinja.palletsprojects.com/en/stable/) is a widely adopted format for LLM prompt
templates. Here are a few examples:

- [`google-gemma-3n-e2b-it.jinja`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/testdata/google-gemma-3n-e2b-it.jinja)
- [`HuggingFaceTB-SmolLM3-3B.jinja`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/testdata/HuggingFaceTB-SmolLM3-3B.jinja)
- [`Qwen-Qwen3-0.6B.jinja`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/testdata/Qwen-Qwen3-0.6B.jinja)

The [Jinja template engine](https://jinja.palletsprojects.com/en/stable/) format should strictly match the structure
expected by the instruction-tuned model. Typically, model releases include the
standard Jinja template to ensure proper model usage.

The Jinja template used by the model will be provided by the model file
metadata.

**Note:** A subtle change in prompt because of incorrect formatting can lead to
significant model degradation. As reported in [Quantifying Language Models'
Sensitivity to Spurious Features in Prompt Design or: How I learned to start
worrying about prompt formatting](https://arxiv.org/abs/2310.11324)

### Preface

[`Preface`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h) sets the initial context for the conversation. It can
include initial messages, tool definitions, and any other background information
the LLM needs to start the interaction. This achieves functionality similar to
the
[`Gemini API system instruction`](https://ai.google.dev/gemini-api/docs/text-generation#system-instructions)
and [`Gemini API Tools`](https://ai.google.dev/api/caching#Tool)

[Preface](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h) contains the following fields

- `messages` The messages in the preface. The messages provided the initial
  background for the conversation. For example, the messages can be the
  conversation history, prompt engineering system instructions, few-shot
  examples, etc.

- `tools` The tools the model can use in the conversation. The format of tools
  is again not fixed, but mostly follows
  [`Gemini API FunctionDeclaration`](https://ai.google.dev/api/caching#FunctionDeclaration).

- `extra_context` The extra context that keeps the extensibility for models to
  customize its required context information to start a conversation. For
  examples,

  - `enable_thinking` for models with thinking mode, e.g. [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) or [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B).

Example preface to provide initial system instruction, tools and disable
thinking mode.

    Preface preface = JsonPreface({
      .messages = {
          {"role", "system"},
          {"content", {"You are a model that can do function calling."}}
        },
      .tools = {
        {
          {"name", "get_weather"},
          {"description", "Returns the weather for a given location."},
          {"parameters", {
            {"type", "object"},
            {"properties", {
              {"location", {
                {"type", "string"},
                {"description", "The location to get the weather for."}
              }}
            }},
            {"required", {"location"}}
          }}
        },
        {
          {"name", "get_stock_price"},
          {"description", "Returns the stock price for a given stock symbol."},
          {"parameters", {
            {"type", "object"},
            {"properties", {
              {"stock_symbol", {
                {"type", "string"},
                {"description", "The stock symbol to get the price for."}
              }}
            }},
            {"required", {"stock_symbol"}}
          }}
        }
      },
      .extra_context = {
        {"enable_thinking": false}
      }
    });

### History

[Conversation](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) maintains a list of all [Message](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h)
exchanges within the session. This history is crucial for prompt template
rendering, as the jinja prompt template typically requires the entire
conversation history to generate the correct prompt for the LLM.

However, the LiteRT-LM [Session](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h) is stateful, meaning it processes
inputs incrementally. To bridge this gap, [Conversation](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) generates
the necessary incremental prompt by rendering the prompt template twice: once
with the history up to the previous turn, and once including the current
message. By comparing these two rendered prompts, it extracts the new portion to
be sent to the [Session](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h).

### ConversationConfig

[`ConversationConfig`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) is used to initialize a
[`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) instance. You can create this configuration in a
couple of ways:

1. **From an [`Engine`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine.h):** This method uses the default [`SessionConfig`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine_settings.h) associated with the engine.
2. **From a specific [`SessionConfig`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/engine/engine_settings.h):** This allows for more fine-grained control over the session settings.

Beyond session settings, you can further customize the
[`Conversation`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) behavior within the
[`ConversationConfig`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h). This includes:

- Providing a [`Preface`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h).
- Overwriting the default [`PromptTemplate`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/components/prompt_template.h).
- Overwriting the default [`DataProcessorConfig`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/model_data_processor/config_registry.h).

These overwrites are particularly useful for fine-tuned models, which might
require different configurations or prompt templates than the base model they
were derived from.

### MessageCallback

[`MessageCallback`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) is the callback function that users should
implement when they use asynchronous [`SendMessageAsync`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h)
method.

The callback signature is `absl::AnyInvocable<void(absl::StatusOr<Message>)>`.
This function is triggered under the following conditions:

- When a new chunk of the [`Message`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h) is received from the Model.
- If an error occurs during LiteRT-LM's message processing.
- Upon completion of the LLM's inference, the callback is triggered with an empty [`Message`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h) (e.g., `JsonMessage()`) to signal the end of the response.

Refer to the [Step 6 asynchronous call](https://developers.google.com/edge/litert-lm/cpp#text-only-content) for an example
implementation.

**Note:** The
[`Message`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/io_types.h)
received by the callback contains only the latest chunk of the model's output,
not the entire message history.

For example, if the complete model response expected from a blocking
[`SendMessage`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) call would be:

    {
      "role": "model",
      "content": [
        "type": "text",
        "text": "Hello World!"
      ]
    }

The callback in [`SendMessageAsync`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/conversation.h) might be invoked multiple
times, each time with a subsequent piece of the text:

    // 1st Message
    {
      "role": "model",
      "content": [
        "type": "text",
        "text": "He"
      ]
    }

    // 2nd Message
    {
      "role": "model",
      "content": [
        "type": "text",
        "text": "llo"
      ]
    }

    // 3rd Message
    {
      "role": "model",
      "content": [
        "type": "text",
        "text": " Wo"
      ]
    }

    // 4th Message
    {
      "role": "model",
      "content": [
        "type": "text",
        "text": "rl"
      ]
    }

    // 5th Message
    {
      "role": "model",
      "content": [
        "type": "text",
        "text": "d!"
      ]
    }

The implementer is responsible for accumulating these chunks if the complete
response is needed during the asynchronous stream. Alternatively, the full
response will be available as the last entry in the [`History`](https://developers.google.com/edge/litert-lm/cpp#history) once
the asynchronous call is complete.

## Advanced Usage

### Constrained Decoding

LiteRT-LM supports constrained decoding, allowing you to enforce specific
structures on the model's output, such as JSON schemas, Regex patterns, or
grammar rules.

To enable it, set `EnableConstrainedDecoding(true)` in `ConversationConfig` and
provide a `ConstraintProviderConfig` (e.g., `LlGuidanceConfig` for
regex/JSON/grammar support). Then, pass constraints via `OptionalArgs` in
`SendMessage`.

**Example: Regex Constraint**

    LlGuidanceConstraintArg constraint_arg;
    constraint_arg.constraint_type = LlgConstraintType::kRegex;
    constraint_arg.constraint_string = "a+b+"; // Force output to match this regex

    auto response = conversation->SendMessage(
        user_message,
        {.decoding_constraint = constraint_arg}
    );

For full details, including JSON Schema and Lark Grammar support, see the [Constrained Decoding documentation](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/cpp/constrained-decoding.md).

### Tool Use

Tool calling allows the LLM to request execution of client-side functions. You
define tools in the `Preface` of the conversation, keying them by name. When
the model outputs a tool call, you capture it, execute the corresponding
function in your application, and return the result to the model.

**High-level flow:**

1. **Declare Tools:** Define tools (name, description, parameters) in the `Preface` JSON.
2. **Detect Calls:** Check `model_message["tool_calls"]` in the response.
3. **Execute:** Run your application logic for the requested tool.
4. **Respond:** Send a message with `role: "tool"` containing the tool's output back to the model.

For full details and a complete chat loop example, see the [Tool Use documentation](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/api/cpp/tool-use.md).