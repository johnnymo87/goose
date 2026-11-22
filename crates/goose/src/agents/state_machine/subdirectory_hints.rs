// Modified by Jonathan Mohrbacher, 2026: subdirectory hints are delivered in the conversation, not the system prompt.
//! Delivers subdirectory hints once, inside the tool response that first touched the
//! directory, instead of folding them into the system prompt.
//!
//! A system prompt that changes whenever a new directory is touched invalidates the
//! provider's prompt cache from the system prompt onward. Delivered hints stay in the
//! conversation, so the prefix never changes for them.
//!
//! Delivery is a pure function of the stored conversation, never of in-memory state.
//! Directories come from agent-visible tool requests, as upstream's system-prompt hints
//! did, and a hint counts as delivered while its block is in an agent-visible tool
//! result. Compaction hides both, so a hint returns with the first call into its
//! directory afterwards, and volume stays bounded by the visible window. Truncation
//! (honk /undo) deletes the block with the calls that found the directory.

use std::path::Path;

use rmcp::model::{Annotations, ContentBlock, Role, TextContent};

use crate::conversation::message::{Message, MessageContent};
use crate::conversation::Conversation;
use crate::hints::get_context_filenames;
use crate::hints::load_hints::SubdirectoryHintTracker;

const HINT_KEY_PREFIX: &str = "subdir_hints:";
const LEAD_IN: &str = "Repository instructions ";

fn marker_open(dir: &str) -> String {
    format!("<subdirectory-hints dir=\"{dir}\">")
}

fn format_hint(dir: &str, content: &str) -> String {
    let lead_in = get_context_filenames()
        .into_iter()
        .find(|name| Path::new(dir).join(name).is_file())
        .map(|name| format!("{LEAD_IN}from {dir}/{name}, which apply to files under {dir}:"))
        .unwrap_or_else(|| format!("{LEAD_IN}that apply to files under {dir}:"));
    format!(
        "{lead_in}\n{}\n{}\n</subdirectory-hints>",
        marker_open(dir),
        content.trim()
    )
}

fn is_hint_block(text: &TextContent) -> bool {
    text.text.starts_with(LEAD_IN)
        && text
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.audience.as_deref())
            == Some(&[Role::Assistant][..])
}

/// Only hint blocks count, so a marker quoted elsewhere (a compaction summary, a
/// transcript grep) does not suppress delivery.
fn message_delivers(message: &Message, marker: &str) -> bool {
    message.content.iter().any(|content| match content {
        MessageContent::ToolResponse(response) => {
            response.tool_result.as_ref().is_ok_and(|result| {
                result.content.iter().any(|block| match block {
                    ContentBlock::Text(text) => is_hint_block(text) && text.text.contains(marker),
                    _ => false,
                })
            })
        }
        _ => false,
    })
}

fn delivered(conversation: &Conversation, dir: &str) -> bool {
    let marker = marker_open(dir);
    conversation
        .messages()
        .iter()
        .filter(|message| message.is_agent_visible())
        .any(|message| message_delivers(message, &marker))
}

fn undelivered_hints(conversation: &Conversation, working_dir: &Path) -> Vec<(String, String)> {
    let mut tracker = SubdirectoryHintTracker::new();
    for message in conversation
        .messages()
        .iter()
        .filter(|message| message.is_agent_visible())
    {
        for content in &message.content {
            if let MessageContent::ToolRequest(request) = content {
                if let Ok(tool_call) = &request.tool_call {
                    tracker.record_tool_arguments(&tool_call.arguments, working_dir);
                }
            }
        }
    }
    let mut hints: Vec<(String, String)> = tracker
        .load_new_hints(working_dir)
        .into_iter()
        .filter_map(|(key, content)| {
            let dir = key
                .strip_prefix(HINT_KEY_PREFIX)
                .unwrap_or(&key)
                .to_string();
            (!delivered(conversation, &dir)).then_some((dir, content))
        })
        .collect();
    hints.sort_by(|a, b| a.0.cmp(&b.0));
    hints
}

/// Appends the hints of directories touched by stored tool requests whose marker is not
/// in an agent-visible message to the last successful tool result in `response`. With no
/// successful result nothing is appended, and a later batch delivers them.
pub(super) fn deliver_subdirectory_hints(
    response: &mut Message,
    conversation: &Conversation,
    working_dir: &Path,
) {
    let Some(result) = response
        .content
        .iter_mut()
        .rev()
        .filter_map(|content| match content {
            MessageContent::ToolResponse(response) => response.tool_result.as_mut().ok(),
            _ => None,
        })
        .find(|result| result.is_error != Some(true))
    else {
        return;
    };
    let hints = undelivered_hints(conversation, working_dir);
    if hints.is_empty() {
        return;
    }
    let text = hints
        .iter()
        .map(|(dir, content)| format_hint(dir, content))
        .collect::<Vec<_>>()
        .join("\n\n");
    result
        .content
        .push(ContentBlock::Text(TextContent::new(text).with_annotations(
            Annotations::default().with_audience(vec![Role::Assistant]),
        )));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::model::{CallToolRequestParams, CallToolResult};
    use std::fs;
    use tempfile::TempDir;

    struct Repo {
        _temp: TempDir,
        root: std::path::PathBuf,
        sub: String,
    }

    fn repo() -> Repo {
        let temp = TempDir::new().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let sub = root.join("sub");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("AGENTS.md"), "sub hint content").unwrap();
        let sub = sub.display().to_string();
        Repo {
            _temp: temp,
            root,
            sub,
        }
    }

    fn read_request(id: &str, path: &str) -> Message {
        let arguments = serde_json::json!({ "path": path })
            .as_object()
            .cloned()
            .unwrap();
        Message::assistant().with_tool_request(
            id,
            Ok(CallToolRequestParams::new("read").with_arguments(arguments)),
        )
    }

    fn tool_response(id: &str, ok: bool) -> Message {
        let result = if ok {
            CallToolResult::success(vec![ContentBlock::text("file body")])
        } else {
            CallToolResult::error(vec![ContentBlock::text("failed")])
        };
        Message::user().with_tool_response(id, Ok(result))
    }

    fn result_texts(message: &Message) -> Vec<String> {
        message
            .content
            .iter()
            .filter_map(|content| match content {
                MessageContent::ToolResponse(response) => response.tool_result.as_ref().ok(),
                _ => None,
            })
            .flat_map(|result| result.content.iter())
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn hints_are_appended_to_the_tool_response_for_the_agent_only() {
        let repo = repo();
        let conversation = Conversation::new_unvalidated([read_request("t1", "sub/file.rs")]);
        let mut response = tool_response("t1", true);

        deliver_subdirectory_hints(&mut response, &conversation, &repo.root);

        let texts = result_texts(&response);
        assert_eq!(texts.len(), 2, "{texts:?}");
        assert!(texts[1].starts_with(&format!(
            "Repository instructions from {}/AGENTS.md, which apply to files under {}:",
            repo.sub, repo.sub
        )));
        assert!(texts[1].contains(&marker_open(&repo.sub)));
        assert!(texts[1].contains("sub hint content"));
        assert!(texts[1].ends_with("</subdirectory-hints>"));
        assert_eq!(
            result_texts(&response.user_visible_content()),
            ["file body"]
        );
    }

    #[test]
    fn hints_already_in_the_conversation_are_not_delivered_again() {
        let repo = repo();
        let mut earlier = tool_response("t1", true);
        deliver_subdirectory_hints(
            &mut earlier,
            &Conversation::new_unvalidated([read_request("t1", "sub/a.rs")]),
            &repo.root,
        );
        let conversation = Conversation::new_unvalidated([
            read_request("t1", "sub/a.rs"),
            earlier,
            read_request("t2", "sub/b.rs"),
        ]);
        let mut response = tool_response("t2", true);

        deliver_subdirectory_hints(&mut response, &conversation, &repo.root);

        assert_eq!(result_texts(&response), ["file body"]);
    }

    #[test]
    fn hints_hidden_from_the_agent_return_with_the_next_call_into_the_directory() {
        let repo = repo();
        let mut earlier = tool_response("t1", true);
        deliver_subdirectory_hints(
            &mut earlier,
            &Conversation::new_unvalidated([read_request("t1", "sub/a.rs")]),
            &repo.root,
        );
        let hidden = [
            read_request("t1", "sub/a.rs").with_visibility(true, false),
            earlier.with_visibility(true, false),
        ];

        let mut elsewhere = tool_response("t2", true);
        let conversation = Conversation::new_unvalidated(
            hidden
                .iter()
                .cloned()
                .chain([read_request("t2", "README.md")]),
        );
        deliver_subdirectory_hints(&mut elsewhere, &conversation, &repo.root);
        assert_eq!(result_texts(&elsewhere), ["file body"]);

        let mut back = tool_response("t3", true);
        let conversation = Conversation::new_unvalidated(
            hidden
                .iter()
                .cloned()
                .chain([read_request("t3", "sub/b.rs")]),
        );
        deliver_subdirectory_hints(&mut back, &conversation, &repo.root);
        let texts = result_texts(&back);
        assert_eq!(texts.len(), 2, "{texts:?}");
        assert!(texts[1].contains(&marker_open(&repo.sub)));
    }

    #[test]
    fn a_quoted_marker_does_not_count_as_delivered() {
        let repo = repo();
        let quoted = Message::user().with_tool_response(
            "t0",
            Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "grep hit: {}",
                marker_open(&repo.sub)
            ))])),
        );
        let conversation = Conversation::new_unvalidated([
            Message::assistant().with_text(marker_open(&repo.sub)),
            quoted,
            read_request("t1", "sub/file.rs"),
        ]);
        let mut response = tool_response("t1", true);

        deliver_subdirectory_hints(&mut response, &conversation, &repo.root);

        assert_eq!(result_texts(&response).len(), 2);
    }

    #[test]
    fn failed_tool_responses_do_not_carry_hints() {
        let repo = repo();
        let conversation = Conversation::new_unvalidated([read_request("t1", "sub/file.rs")]);
        let mut response = tool_response("t1", false);

        deliver_subdirectory_hints(&mut response, &conversation, &repo.root);

        assert_eq!(result_texts(&response), ["failed"]);
    }

    #[test]
    fn hints_go_to_the_last_successful_response_in_directory_order() {
        let repo = repo();
        let other = repo.root.join("other");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("AGENTS.md"), "other hint content").unwrap();
        let conversation = Conversation::new_unvalidated([
            read_request("t1", "sub/file.rs"),
            read_request("t2", "other/file.rs"),
        ]);
        let mut response = Message::user()
            .with_tool_response(
                "t1",
                Ok(CallToolResult::success(vec![ContentBlock::text("one")])),
            )
            .with_tool_response(
                "t2",
                Ok(CallToolResult::error(vec![ContentBlock::text("two")])),
            );

        deliver_subdirectory_hints(&mut response, &conversation, &repo.root);

        let texts = result_texts(&response);
        assert_eq!(texts.len(), 3, "{texts:?}");
        assert_eq!(texts[0], "one");
        assert_eq!(texts[2], "two");
        let hints = &texts[1];
        let other_at = hints
            .find(&marker_open(&other.display().to_string()))
            .unwrap();
        let sub_at = hints.find(&marker_open(&repo.sub)).unwrap();
        assert!(other_at < sub_at, "{hints}");
    }
}
