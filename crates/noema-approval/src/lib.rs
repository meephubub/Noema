//! Human approval infrastructure for Noema.
//!
//! Tools declare a [`RiskLevel`]; the session's [`ApprovalPolicy`] decides
//! which calls must pause for a human. A paused call becomes an
//! [`ApprovalRequest`] held in an [`ApprovalStore`]; the frontend sees the
//! complete proposal (tool, arguments, risk, expiry) through the
//! `ToolApprovalRequired` event and answers with `Approve` or `Reject`
//! through `session.approve_tool` / `session.reject_tool`.
//!
//! ```text
//! tool call
//!     ↓
//! risk evaluation (ApprovalPolicy)
//!     ├── below threshold → execute immediately
//!     └── at/above threshold → ApprovalRequest (ToolApprovalRequired)
//!              ├── approved → execute
//!              ├── rejected → cancelled
//!              └── timeout → expired
//! ```
//!
//! The policy is enforced by Noema, never by the frontend and never by the
//! model, and [`RiskLevel::Critical`] always requires approval.

pub mod error;
pub mod policy;
pub mod request;
pub mod store;

pub use error::{ApprovalError, Result};
pub use policy::ApprovalPolicy;
pub use request::{
    ApprovalDecision, ApprovalHandle, ApprovalId, ApprovalRequest, ApprovalStatus,
};
pub use store::{ApprovalStore, SharedApprovalStore};
