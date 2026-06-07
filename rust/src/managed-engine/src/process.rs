use std::io;
use std::net::TcpListener;
use std::process::{Command as StdCommand, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tokio::time::interval;
use tracing::info;

const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(200);
const MIN_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Allocate one ephemeral TCP port for the managed headless-engine handshake on
/// the given host.
pub fn allocate_handshake_port(host: &str) -> Result<u16> {
    let listener = TcpListener::bind((host, 0)).context("failed to allocate handshake port")?;
    let port = listener
        .local_addr()
        .context("failed to inspect allocated handshake listener address")?
        .port();
    Ok(port)
}

/// Spawn configuration for one managed headless Python vLLM engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedEngineConfig {
    /// Python executable used to launch `vllm.entrypoints.cli.main`.
    pub python: String,
    /// Model identifier passed to `vllm ... serve <model>`.
    pub model: String,
    /// Host portion of the headless-engine handshake endpoint.
    pub handshake_host: String,
    /// Port portion of the headless-engine handshake endpoint.
    pub handshake_port: u16,
    /// Number of data parallel replicas across the whole deployment.
    ///
    /// The per-node replica count is forwarded separately in `python_args` as
    /// `--data-parallel-size-local`.
    pub data_parallel_size: usize,
    /// Extra CLI arguments forwarded verbatim to Python vLLM.
    pub python_args: Vec<String>,
}

impl ManagedEngineConfig {
    /// Render the handshake address that the Rust frontend should dial.
    pub fn handshake_address(&self) -> String {
        format!("tcp://{}:{}", self.handshake_host, self.handshake_port)
    }

    /// Build the concrete Python command line for the managed headless engine.
    pub fn to_command(&self) -> StdCommand {
        let mut command = StdCommand::new(&self.python);
        command
            .arg("-m")
            .arg("vllm.entrypoints.cli.main")
            .arg("serve")
            .arg(&self.model)
            .arg("--headless")
            .arg("--data-parallel-address")
            .arg(&self.handshake_host)
            .arg("--data-parallel-rpc-port")
            .arg(self.handshake_port.to_string())
            .arg("--data-parallel-size")
            .arg(self.data_parallel_size.to_string())
            .args(&self.python_args);
        command
    }
}

/// RAII-style handle for one managed Python headless engine subprocess.
#[derive(Clone)]
pub struct ManagedEngineHandle {
    child: Arc<Mutex<Child>>,
    shutdown_started: Arc<AtomicBool>,
    /// Windows Job Object owning the engine subtree. Dropping it (via the last
    /// `Arc`) tears down the whole tree thanks to `KILL_ON_JOB_CLOSE`, which is
    /// the closest analog to a POSIX process group. `None` if the job could not
    /// be created, in which case shutdown falls back to killing the immediate
    /// child only.
    #[cfg(windows)]
    job: Arc<Option<process_group::JobObject>>,
}

impl ManagedEngineHandle {
    /// Spawn one managed Python headless engine and return a handle for
    /// monitoring it.
    pub async fn spawn(config: ManagedEngineConfig) -> Result<Self> {
        let command = config.to_command();
        info!(
            handshake_address = %config.handshake_address(),
            ?command,
            "starting managed Python headless engine"
        );

        let mut command = Command::from(command);
        command.stdin(Stdio::null()).stdout(Stdio::inherit()).stderr(Stdio::inherit());

        process_group::configure(&mut command);

        let child = command.spawn().context("failed to spawn managed engine")?;

        // On Windows, assign the freshly spawned child to a kill-on-close job
        // object before it has a chance to spawn its own workers. This must
        // happen while we still have a borrow of `child`, before it is moved
        // into the handle below.
        #[cfg(windows)]
        let job = Arc::new(process_group::assign_to_new_job(&child));

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            shutdown_started: Arc::new(AtomicBool::new(false)),
            #[cfg(windows)]
            job,
        })
    }

    /// Poll whether the managed engine has exited yet.
    pub async fn try_wait(&self) -> Option<ExitStatus> {
        let mut child = self.child.lock().await;
        child.try_wait().expect("failed to poll the status of managed engine")
    }

    /// Wait until the managed engine exits.
    pub async fn wait_for_exit(&self) -> ExitStatus {
        let mut interval = interval(CHILD_POLL_INTERVAL);
        loop {
            interval.tick().await;
            if let Some(status) = self.try_wait().await {
                return status;
            }
        }
    }

    /// Terminate the managed engine process group and wait for it to stop.
    pub async fn shutdown(&self, timeout: Duration) -> Result<()> {
        if self.shutdown_started.swap(true, Ordering::SeqCst) {
            return Ok(());
        }

        let Some(pid) = self.child.lock().await.id() else {
            return Ok(());
        };

        // Enforce a minimum shutdown timeout to give the engine process enough time to
        // clean up.
        let shutdown_timeout = std::cmp::max(timeout, MIN_SHUTDOWN_TIMEOUT);

        // First, try to gracefully terminate.
        #[cfg(unix)]
        info!(
            pid,
            ?shutdown_timeout,
            "shutting down managed engine with SIGTERM"
        );
        #[cfg(windows)]
        info!(
            pid,
            ?shutdown_timeout,
            "shutting down managed engine with Ctrl-Break"
        );
        #[cfg(unix)]
        process_group::terminate(pid)?;
        #[cfg(windows)]
        process_group::terminate(pid)?;

        // Wait for the process to exit on its own.
        if tokio::time::timeout(shutdown_timeout, self.wait_for_exit()).await.is_ok() {
            return Ok(());
        }

        // If it doesn't exit within the timeout, force kill it.
        #[cfg(unix)]
        info!(
            pid,
            "managed engine did not exit within timeout, sending SIGKILL"
        );
        #[cfg(windows)]
        info!(
            pid,
            "managed engine did not exit within timeout, terminating job object"
        );
        #[cfg(unix)]
        process_group::kill(pid)?;
        #[cfg(windows)]
        {
            // Kill the immediate child directly, then terminate the job object
            // (if any) to take down worker grandchildren as well.
            let _ = self.child.lock().await.start_kill();
            if let Some(job) = (*self.job).as_ref() {
                process_group::kill(job)?;
            }
        }

        let _ = self.wait_for_exit().await;
        Ok(())
    }
}

/// Process group helper functions for managing the Python subprocess and its
/// potential children in a platform-aware way.
mod process_group {
    use super::*;

    /// Place the Python child into its own process group so `serve` can tear
    /// down the whole subtree rather than just the immediate shell process.
    #[cfg(unix)]
    pub fn configure(command: &mut Command) {
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }

    /// Send SIGTERM to the managed Python process group.
    #[cfg(unix)]
    pub fn terminate(pid: u32) -> Result<()> {
        signal(pid, libc::SIGTERM)
    }

    /// Send SIGKILL to the managed Python process group.
    #[cfg(unix)]
    pub fn kill(pid: u32) -> Result<()> {
        signal(pid, libc::SIGKILL)
    }

    /// Deliver one signal to the managed Python process group.
    #[cfg(unix)]
    fn signal(pid: u32, signal: i32) -> Result<()> {
        let rc = unsafe { libc::kill(-(pid as i32), signal) };
        if rc == 0 {
            return Ok(());
        }

        let error = io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(code) if code == libc::ESRCH) {
            return Ok(());
        }
        Err(error).context("failed to signal managed engine process group")
    }

    // --- Windows implementation ------------------------------------------------

    /// A Windows Job Object owning the managed engine subtree. Configured with
    /// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so dropping it (closing the
    /// handle) terminates every process still assigned to the job — the engine
    /// and any worker grandchildren it spawned. This is the closest analog to a
    /// POSIX process group.
    #[cfg(windows)]
    pub struct JobObject(std::os::windows::io::OwnedHandle);

    // SAFETY: a job object handle is just a kernel handle; it is safe to send
    // and share across threads (all access is through the Win32 API which is
    // internally synchronized).
    #[cfg(windows)]
    unsafe impl Send for JobObject {}
    #[cfg(windows)]
    unsafe impl Sync for JobObject {}

    /// Put the child into a new process group so a console control event can be
    /// delivered to the whole subtree (the Windows analog of placing the child
    /// in its own POSIX process group).
    #[cfg(windows)]
    pub fn configure(command: &mut Command) {
        use windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }

    /// Create a kill-on-close job object and assign the spawned child to it.
    ///
    /// Returns `None` if the job object could not be created; the caller then
    /// falls back to killing only the immediate child on shutdown. Assignment
    /// failures are logged but non-fatal.
    #[cfg(windows)]
    pub fn assign_to_new_job(child: &Child) -> Option<JobObject> {
        use std::os::windows::io::{FromRawHandle, OwnedHandle, RawHandle};

        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject,
        };

        // SAFETY: all calls below follow the documented Win32 Job Object
        // contract; `handle` is checked for null before being adopted by an
        // `OwnedHandle` that closes it on drop.
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                tracing::warn!(
                    error = %io::Error::last_os_error(),
                    "failed to create job object for managed engine; \
                     grandchild cleanup is not guaranteed"
                );
                return None;
            }
            let job = OwnedHandle::from_raw_handle(handle as RawHandle);

            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(info) as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                tracing::warn!(
                    error = %io::Error::last_os_error(),
                    "failed to set kill-on-close limit on managed engine job object"
                );
            }

            match child.raw_handle() {
                Some(raw) => {
                    if AssignProcessToJobObject(handle, raw as HANDLE) == 0 {
                        tracing::warn!(
                            error = %io::Error::last_os_error(),
                            "failed to assign managed engine to job object; \
                             grandchild cleanup is not guaranteed"
                        );
                    }
                }
                None => tracing::warn!(
                    "managed engine exited before it could be assigned to a job object"
                ),
            }

            Some(JobObject(job))
        }
    }

    /// Request graceful shutdown by sending a Ctrl-Break console event to the
    /// managed engine's process group (its pid, since it was created with
    /// `CREATE_NEW_PROCESS_GROUP`). Best effort: failures (e.g. no attached
    /// console) are ignored so the caller proceeds to a forced kill.
    #[cfg(windows)]
    pub fn terminate(pid: u32) -> Result<()> {
        use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, GenerateConsoleCtrlEvent};

        // SAFETY: a plain Win32 call with a process-group id and event code.
        unsafe {
            if GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid) == 0 {
                tracing::debug!(
                    error = %io::Error::last_os_error(),
                    "could not deliver Ctrl-Break to managed engine; \
                     will fall back to terminating the job object"
                );
            }
        }
        Ok(())
    }

    /// Forcefully terminate every process assigned to the job object.
    #[cfg(windows)]
    pub fn kill(job: &JobObject) -> Result<()> {
        use std::os::windows::io::AsRawHandle;

        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        // SAFETY: `job` owns a valid job object handle for the duration of the
        // borrow.
        unsafe {
            if TerminateJobObject(job.0.as_raw_handle() as HANDLE, 1) == 0 {
                let error = io::Error::last_os_error();
                return Err(error).context("failed to terminate managed engine job object");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use super::{ManagedEngineConfig, allocate_handshake_port};

    #[test]
    fn command_snapshot() {
        let config = ManagedEngineConfig {
            python: "python3".to_string(),
            model: "Qwen/Qwen3-0.6B".to_string(),
            handshake_host: "127.0.0.1".to_string(),
            handshake_port: 62100,
            data_parallel_size: 4,
            python_args: vec![
                "--data-parallel-size-local".to_string(),
                "2".to_string(),
                "--data-parallel-start-rank".to_string(),
                "2".to_string(),
                "--dtype".to_string(),
                "float16".to_string(),
                "--max-model-len".to_string(),
                "512".to_string(),
            ],
        };
        let command = config.to_command();
        let args = command.get_args().collect::<Vec<_>>();

        expect![[r#"
            [
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                "Qwen/Qwen3-0.6B",
                "--headless",
                "--data-parallel-address",
                "127.0.0.1",
                "--data-parallel-rpc-port",
                "62100",
                "--data-parallel-size",
                "4",
                "--data-parallel-size-local",
                "2",
                "--data-parallel-start-rank",
                "2",
                "--dtype",
                "float16",
                "--max-model-len",
                "512",
            ]
        "#]]
        .assert_debug_eq(&args);
    }

    #[test]
    fn allocate_handshake_port_returns_non_zero_port() {
        let port = allocate_handshake_port("127.0.0.1").unwrap();
        assert_ne!(port, 0);
    }
}
