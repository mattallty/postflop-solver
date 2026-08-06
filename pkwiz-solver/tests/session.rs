//! The session contract.
//!
//! **A command can fail; a session cannot.** These tests are the enforcement of that sentence:
//! garbage in, a command that does not exist, a job that does not exist, and a panic in the
//! engine all have to come back as one JSON frame with the session still answering. A solver
//! session holds every solved tree in memory, so losing it to a bad line is considerably more
//! expensive than losing a parse session.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pkwiz_solver::{Collector, Command, Emit, OpError, Session};

fn session() -> (Session, Arc<Collector>) {
    let collector = Arc::new(Collector::default());
    let session = Session::new(Arc::clone(&collector) as Arc<dyn Emit>);
    (session, collector)
}

fn call(session: &Session, line: &str) -> serde_json::Value {
    let handled = session.handle_line(line);
    serde_json::from_str(&pkwiz_solver::encode(&handled.response))
        .expect("every response is valid JSON")
}

/// A river spot small enough that a test can wait for it: nine combinations, thirty iterations.
const SPOT: &str = r#"{"oop":"QQ+","ip":"QQ+","board":"2c7dTh4sQd","pot":100,"effectiveStack":100,"stop":{"maxIterations":30,"checkInterval":5}}"#;

#[test]
fn malformed_and_unknown_requests_are_bad_request() {
    let (session, _) = session();
    for line in [
        "not json at all",
        "{}",
        "[]",
        r#"{"cmd":"noSuchCommand"}"#,
        r#"{"cmd":"solve"}"#,                      // missing `spot`
        r#"{"cmd":"solve","spot":7}"#,             // wrong type
        r#"{"cmd":"progress"}"#,                   // missing `jobId`
        r#"{"cmd":"progress","jobId":"one"}"#,     // wrong type
        r#"{"cmd":"solve","spot":{"oop":"QQ+"}}"#, // half a spot
    ] {
        let v = call(&session, line);
        assert_eq!(v["ok"], false, "{line} should be rejected");
        assert_eq!(v["error"]["code"], "bad_request", "{line}");
        assert!(v["result"].is_null());
    }
}

#[test]
fn a_spot_the_engine_would_reject_is_rejected_synchronously() {
    let (session, _) = session();

    // A bad range, a bad board and a bad bet size are all form errors: they come back on the
    // `solve` command itself, not as a job that fails a moment later, so a UI can point at the
    // field. The code stays `engine` because the host branches on that, not on the prose.
    for (line, needle) in [
        (
            r#"{"id":1,"cmd":"solve","spot":{"oop":"XX","ip":"QQ+","board":"2c7dTh","pot":1,"effectiveStack":1}}"#,
            "XX",
        ),
        (
            r#"{"id":2,"cmd":"solve","spot":{"oop":"QQ+","ip":"QQ+","board":"2c7d","pot":1,"effectiveStack":1}}"#,
            "three, four or five",
        ),
        (
            r#"{"id":3,"cmd":"solve","spot":{"oop":"QQ+","ip":"QQ+","board":"2c7dTh","pot":0,"effectiveStack":1}}"#,
            "pot",
        ),
        (
            r#"{"id":4,"cmd":"solve","spot":{"oop":"QQ+","ip":"QQ+","board":"2c7dTh","pot":1,"effectiveStack":1,
                "sizing":{"flop":{"bet":"banana","raise":"2.5x"}}}}"#,
            "banana",
        ),
    ] {
        let v = call(&session, line);
        assert_eq!(v["ok"], false, "{line}");
        assert_eq!(v["error"]["code"], "engine", "{line}");
        assert!(
            v["error"]["message"].as_str().unwrap().contains(needle),
            "{}",
            v["error"]["message"]
        );
    }
}

#[test]
fn an_unknown_job_is_reported_not_fatal() {
    let (session, _) = session();
    for line in [
        r#"{"id":1,"cmd":"progress","jobId":404}"#,
        r#"{"id":2,"cmd":"cancel","jobId":404}"#,
        r#"{"id":3,"cmd":"node","jobId":404}"#,
        r#"{"id":4,"cmd":"save","jobId":404,"path":"/tmp/nope.bin"}"#,
    ] {
        let v = call(&session, line);
        assert_eq!(v["ok"], false, "{line}");
        assert_eq!(v["error"]["code"], "no_such_job", "{line}");
    }
    // Still alive.
    assert_eq!(call(&session, r#"{"cmd":"ping"}"#)["result"]["pong"], true);
}

#[test]
fn opening_a_file_that_is_not_a_solution_is_an_error_not_a_crash() {
    let (session, _) = session();
    let v = call(
        &session,
        r#"{"id":1,"cmd":"open","path":"/definitely/not/here.bin"}"#,
    );
    assert_eq!(v["ok"], false);
    assert_eq!(v["error"]["code"], "engine");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("/definitely/not/here.bin"));

    // A real file that is not a solution: the engine's own reader has to reject it, and its
    // complaint has to arrive as JSON rather than as a torn-off session.
    let path = std::env::temp_dir().join(format!("pkwiz-not-a-solution-{}", std::process::id()));
    std::fs::write(&path, b"this is not a bincode game tree").unwrap();
    let v = call(
        &session,
        &format!(
            r#"{{"id":2,"cmd":"open","path":{}}}"#,
            serde_json::to_string(&path.to_string_lossy()).unwrap()
        ),
    );
    assert_eq!(v["ok"], false);
    assert_eq!(v["error"]["code"], "engine");
    std::fs::remove_file(&path).ok();

    assert_eq!(call(&session, r#"{"cmd":"ping"}"#)["result"]["pong"], true);
}

#[test]
fn a_panic_in_the_engine_becomes_a_response_not_a_dead_session() {
    // The engine panics rather than returning an error for several kinds of misuse and is full of
    // `unsafe` in its hot loops, so this path is not hypothetical. The injected executor stands in
    // for that; what matters is that the session still answers afterwards.
    let (session, _) = session();

    let handled = session.handle_line_with(r#"{"id":3,"cmd":"ping"}"#, |_| {
        panic!("simulated engine bug")
    });
    assert!(!handled.stop);
    let body = handled.response.error.expect("a panic must be reported");
    assert_eq!(body.code, "panic");
    assert!(body.message.contains("simulated engine bug"), "{body:?}");
    assert_eq!(handled.response.id, Some(serde_json::json!(3)));

    let handled = session.handle_line_with(r#"{"id":4,"cmd":"ping"}"#, |_| {
        std::panic::panic_any(String::from("owned payload"))
    });
    assert!(handled
        .response
        .error
        .expect("reported")
        .message
        .contains("owned payload"));

    assert_eq!(call(&session, r#"{"cmd":"ping"}"#)["result"]["pong"], true);
}

#[test]
fn version_reports_the_engine_it_was_built_against() {
    let (session, _) = session();
    let v = call(&session, r#"{"id":1,"cmd":"version"}"#);
    assert_eq!(v["result"]["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(
        v["result"]["protocolVersion"],
        pkwiz_solver::PROTOCOL_VERSION
    );
    // Pinned to the literal: bunching arrived as additive changes only — new commands, new
    // optional fields — so the protocol version must not have moved, and neither must the
    // engine pin (the whole feature is sidecar-side; `tests/engine_rev.rs` enforces the rest).
    assert_eq!(v["result"]["protocolVersion"], 1);
    // A saved solution is only readable by an engine that writes the same format, so the pin is
    // part of the handshake and not a decoration.
    assert_eq!(v["result"]["engineRev"], pkwiz_solver::protocol::ENGINE_REV);
    assert_eq!(
        v["result"]["engineFormat"],
        pkwiz_solver::protocol::ENGINE_FORMAT
    );

    // The compatibility list is what a host actually decides readability on, so it travels too,
    // and it always contains the revision we are built on.
    let revs = v["result"]["engineCompatibleRevs"]
        .as_array()
        .expect("a list of revisions");
    assert!(
        revs.iter().any(|r| r == pkwiz_solver::protocol::ENGINE_REV),
        "{revs:?}"
    );
    assert_eq!(
        revs.len(),
        pkwiz_solver::protocol::ENGINE_COMPATIBLE_REVS.len()
    );
}

#[test]
fn releasing_a_tree_over_the_wire_keeps_the_job_and_its_numbers() {
    let (session, _) = session();
    let dir = std::env::temp_dir().join(format!("pkwiz-session-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("wire-release.bin");
    let path_text = path.to_string_lossy().into_owned();

    let id = call(&session, &format!(r#"{{"cmd":"solve","spot":{SPOT}}}"#))["result"]["jobId"]
        .as_u64()
        .unwrap();

    // Wait for it, then write it out — a job with no file behind it cannot be released.
    let deadline = Instant::now() + Duration::from_secs(30);
    let done = loop {
        let v = call(&session, &format!(r#"{{"cmd":"progress","jobId":{id}}}"#));
        if v["result"]["phase"] == "done" {
            break v;
        }
        assert!(Instant::now() < deadline, "job never finished: {v}");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    assert_eq!(done["result"]["resident"], true);

    let saved = call(
        &session,
        &format!(
            r#"{{"cmd":"save","jobId":{id},"path":{}}}"#,
            serde_json::json!(path_text)
        ),
    );
    assert_eq!(saved["ok"], true, "{saved:#?}");

    let released = call(&session, &format!(r#"{{"cmd":"release","jobId":{id}}}"#));
    assert_eq!(released["ok"], true, "{released:#?}");
    assert_eq!(released["result"]["resident"], false);
    assert_eq!(released["result"]["phase"], "done");
    assert_eq!(released["result"]["ev"], done["result"]["ev"]);

    // Still browsable: the file is reopened on demand.
    let node = call(&session, &format!(r#"{{"cmd":"node","jobId":{id}}}"#));
    assert_eq!(node["ok"], true, "{node:#?}");
    assert_eq!(
        call(&session, &format!(r#"{{"cmd":"progress","jobId":{id}}}"#))["result"]["resident"],
        true
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn shutdown_answers_before_it_stops() {
    let (session, _) = session();
    let handled = session.handle_line(r#"{"id":9,"cmd":"shutdown"}"#);
    assert!(handled.stop);
    assert!(handled.response.ok);
    assert_eq!(handled.response.id, Some(serde_json::json!(9)));
}

#[test]
fn a_session_pipelines_and_keeps_going_after_a_bad_line() {
    let (session, collector) = session();
    let input = format!(
        "{}\n\ngarbage\n{}\n{}\n",
        r#"{"id":1,"cmd":"ping"}"#,
        format_args!(r#"{{"id":2,"cmd":"solve","spot":{SPOT}}}"#),
        r#"{"id":3,"cmd":"shutdown"}"#,
    );

    session
        .run(std::io::Cursor::new(input))
        .expect("a session must not fail");

    // Responses in request order, interleaved with however many job events the solve produced.
    let responses: Vec<serde_json::Value> = collector
        .frames()
        .into_iter()
        .filter(|f| f.get("event").is_none())
        .collect();
    assert_eq!(responses.len(), 4, "{responses:#?}");
    assert_eq!(responses[0]["id"], 1);
    assert_eq!(responses[1]["error"]["code"], "bad_request");
    assert_eq!(responses[1]["id"], serde_json::Value::Null);
    assert_eq!(responses[2]["id"], 2);
    assert_eq!(responses[2]["result"]["phase"], "queued");
    assert_eq!(responses[3]["id"], 3);
}

#[test]
fn events_and_responses_are_told_apart_by_a_key_not_by_luck() {
    let (session, collector) = session();
    let queued = call(
        &session,
        &format!(r#"{{"id":1,"cmd":"solve","spot":{SPOT}}}"#),
    );
    let id = queued["result"]["jobId"].as_u64().unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let v = call(&session, &format!(r#"{{"cmd":"progress","jobId":{id}}}"#));
        if v["result"]["phase"] == "done" {
            break;
        }
        assert!(Instant::now() < deadline, "never finished: {v}");
        std::thread::sleep(Duration::from_millis(5));
    }

    let events = collector.events();
    assert!(!events.is_empty(), "the solve pushed nothing");
    for event in &events {
        assert_eq!(event["event"], "job");
        assert!(
            event.get("id").is_none() && event.get("ok").is_none(),
            "an event must not look like a response: {event}"
        );
        assert_eq!(event["job"]["jobId"], id);
    }
    // The last event and a poll agree, because they are the same struct.
    let polled = call(&session, &format!(r#"{{"cmd":"progress","jobId":{id}}}"#));
    assert_eq!(events.last().unwrap()["job"], polled["result"]);
}

#[test]
fn a_frame_never_contains_a_raw_newline() {
    // The framing is the protocol. Error messages quote user input, and user input arrives from a
    // text box that can contain anything.
    let (session, _) = session();
    let handled = session.handle_line(
        r#"{"cmd":"solve","spot":{"oop":"line\none,line\ntwo","ip":"QQ+","board":"2c7dTh","pot":1,"effectiveStack":1}}"#,
    );
    let encoded = pkwiz_solver::encode(&handled.response);
    assert!(!handled.response.ok);
    assert!(!encoded.contains('\n'));
    assert!(!encoded.contains('\r'));
}

#[test]
fn the_job_list_grows_and_a_node_needs_a_finished_job() {
    let (session, _) = session();
    assert_eq!(
        call(&session, r#"{"cmd":"jobs"}"#)["result"]
            .as_array()
            .unwrap()
            .len(),
        0
    );

    let id = call(&session, &format!(r#"{{"cmd":"solve","spot":{SPOT}}}"#))["result"]["jobId"]
        .as_u64()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let v = call(&session, &format!(r#"{{"cmd":"node","jobId":{id}}}"#));
        if v["ok"] == true {
            assert_eq!(v["result"]["player"], 0);
            assert!(!v["result"]["hands"].as_array().unwrap().is_empty());
            assert_eq!(v["result"]["board"][0], "2c");
            break;
        }
        assert_eq!(v["error"]["code"], "not_readable", "{v}");
        assert!(Instant::now() < deadline, "never finished");
        std::thread::sleep(Duration::from_millis(5));
    }

    assert_eq!(
        call(&session, r#"{"cmd":"jobs"}"#)["result"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn estimate_answers_without_solving() {
    let (session, _) = session();
    let v = call(
        &session,
        r#"{"id":1,"cmd":"estimate","spot":{"oop":"22+","ip":"22+","board":"2c7dTh","pot":100,"effectiveStack":300}}"#,
    );
    assert_eq!(v["ok"], true, "{v}");
    let uncompressed = v["result"]["uncompressed"].as_u64().unwrap();
    assert!(uncompressed > 0);
    assert!(v["result"]["compressed"].as_u64().unwrap() < uncompressed);
    // Nothing was queued.
    assert_eq!(
        call(&session, r#"{"cmd":"jobs"}"#)["result"]
            .as_array()
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn the_executor_is_reachable_without_a_session() {
    // Nothing links this crate as a library — the process boundary is the point, and linking it
    // would take the engine's licence with it. The layering is kept anyway, so the protocol can
    // be exercised without standing up a transport.
    let jobs = pkwiz_solver::Jobs::new(Arc::new(pkwiz_solver::Silent));
    let value = pkwiz_solver::execute(&jobs, Command::Ping).unwrap();
    assert_eq!(value["pong"], true);

    let err = pkwiz_solver::execute(&jobs, Command::Progress { job_id: 1 }).unwrap_err();
    assert!(matches!(err, OpError::Job(_)));
    assert_eq!(err.code(), "no_such_job");
}

#[test]
fn a_prep_and_its_solve_pipeline_in_one_breath() {
    // The FIFO promise a host leans on: submit the preparation and the solve that uses it
    // back-to-back, without reading anything in between, and the preparation is guaranteed to
    // have finished before the solve builds. In a fresh session the first job id is 1, which is
    // what lets the second line be written blind.
    let (session, collector) = session();

    let prep = call(
        &session,
        r#"{"id":1,"cmd":"prepareBunching","bunching":{"foldRanges":["AA"],"flop":"2c7dTh"}}"#,
    );
    assert_eq!(prep["ok"], true, "{prep}");
    assert_eq!(prep["result"]["jobId"], 1);
    assert_eq!(prep["result"]["kind"], "bunching");
    assert_eq!(prep["result"]["phase"], "queued");
    assert_eq!(prep["result"]["bunching"]["foldPlayers"], 1);

    let solve = call(
        &session,
        &format!(r#"{{"id":2,"cmd":"solve","spot":{SPOT_WITH_BUNCHING}}}"#),
    );
    assert_eq!(solve["ok"], true, "{solve}");
    assert_eq!(solve["result"]["kind"], "solve");
    let solve_id = solve["result"]["jobId"].as_u64().unwrap();

    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let v = call(
            &session,
            &format!(r#"{{"cmd":"progress","jobId":{solve_id}}}"#),
        );
        if v["result"]["phase"] == "done" {
            break;
        }
        assert!(
            v["result"]["phase"] != "failed",
            "the solve failed: {}",
            v["result"]["error"]
        );
        assert!(Instant::now() < deadline, "never finished: {v}");
        std::thread::sleep(Duration::from_millis(5));
    }

    // Every job frame says what kind of job it is, and events stay distinguishable from
    // responses by the `event` key alone.
    let events = collector.events();
    for event in &events {
        assert_eq!(event["event"], "job");
        assert!(event.get("id").is_none() && event.get("ok").is_none());
        assert!(
            event["job"]["kind"] == "bunching" || event["job"]["kind"] == "solve",
            "a frame without a kind: {event}"
        );
    }
    // FIFO in the frames themselves: the preparation's terminal frame precedes the solve's
    // first building frame.
    let prep_done = events
        .iter()
        .position(|e| e["job"]["jobId"] == 1 && e["job"]["phase"] == "done")
        .expect("the preparation finished");
    let solve_building = events
        .iter()
        .position(|e| e["job"]["jobId"] == solve_id && e["job"]["phase"] == "building")
        .expect("the solve built");
    assert!(
        prep_done < solve_building,
        "{prep_done} vs {solve_building}"
    );
}

/// The bunching pipeline's solve half: same flop as the preparation above, referencing job 1.
const SPOT_WITH_BUNCHING: &str = r#"{"oop":"KK","ip":"AA,JJ","board":"2c7dTh4sQd","pot":100,"effectiveStack":100,"stop":{"maxIterations":20,"checkInterval":5},"bunching":{"jobId":1}}"#;

#[test]
fn a_line_that_is_not_utf8_is_a_bad_request_not_the_end_of_the_session() {
    // `BufRead::lines` would answer one stray byte with `InvalidData`, ending the loop — and
    // with it the queue and every in-memory tree. The contract says a command can fail and a
    // session cannot, so the corrupt line gets a response and the next line gets served.
    let (session, collector) = session();

    let mut input: Vec<u8> = Vec::new();
    input.extend_from_slice(b"\xff\xfe{\"cmd\":\"ping\"}\n");
    input.extend_from_slice(b"{\"id\":7,\"cmd\":\"ping\"}\n");
    session
        .run(std::io::BufReader::new(input.as_slice()))
        .expect("a corrupt line is not an I/O error");

    let frames = collector.frames();
    assert_eq!(frames.len(), 2, "{frames:?}");
    assert_eq!(frames[0]["ok"], false);
    assert_eq!(frames[0]["error"]["code"], "bad_request");
    assert_eq!(
        frames[1]["ok"], true,
        "the session kept serving: {frames:?}"
    );
    assert_eq!(frames[1]["id"], 7);
}

#[test]
fn an_oversized_line_is_discarded_not_buffered_without_bound() {
    // A host that never sends its newline must not grow this process's line buffer forever;
    // past the cap the line is swallowed in bounded chunks and answered like any bad request.
    let (session, collector) = session();

    let mut input = vec![b'x'; pkwiz_solver::MAX_LINE_BYTES + 4096];
    input.push(b'\n');
    input.extend_from_slice(b"{\"id\":8,\"cmd\":\"ping\"}\n");
    session
        .run(std::io::BufReader::new(input.as_slice()))
        .expect("an oversized line is not an I/O error");

    let frames = collector.frames();
    assert_eq!(frames.len(), 2, "{frames:?}");
    assert_eq!(frames[0]["ok"], false);
    assert_eq!(frames[0]["error"]["code"], "bad_request");
    assert_eq!(frames[1]["ok"], true);
    assert_eq!(frames[1]["id"], 8);
}

#[test]
fn forget_over_the_wire_removes_the_job() {
    let (session, _) = session();
    let queued = call(
        &session,
        &format!(r#"{{"id":1,"cmd":"solve","spot":{SPOT}}}"#),
    );
    let id = queued["result"]["jobId"].as_u64().unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let v = call(
            &session,
            &format!(r#"{{"id":2,"cmd":"progress","jobId":{id}}}"#),
        );
        if v["result"]["phase"] == "done" {
            break;
        }
        assert!(Instant::now() < deadline, "job never finished: {v}");
        std::thread::sleep(Duration::from_millis(5));
    }

    let forgotten = call(
        &session,
        &format!(r#"{{"id":3,"cmd":"forget","jobId":{id}}}"#),
    );
    assert_eq!(forgotten["ok"], true, "{forgotten}");
    assert_eq!(forgotten["result"]["phase"], "done");

    let after = call(
        &session,
        &format!(r#"{{"id":4,"cmd":"progress","jobId":{id}}}"#),
    );
    assert_eq!(after["ok"], false);
    assert_eq!(after["error"]["code"], "no_such_job");
}

#[test]
fn a_node_response_carries_ev_detail_and_lock_state_in_camel_case() {
    let (session, _) = session();
    let queued = call(
        &session,
        &format!(r#"{{"id":1,"cmd":"solve","spot":{SPOT}}}"#),
    );
    let id = queued["result"]["jobId"].as_u64().unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let v = call(&session, &format!(r#"{{"cmd":"progress","jobId":{id}}}"#));
        if v["result"]["phase"] == "done" {
            break;
        }
        assert!(Instant::now() < deadline, "job never finished: {v}");
        std::thread::sleep(Duration::from_millis(5));
    }

    let node = call(
        &session,
        &format!(r#"{{"id":2,"cmd":"node","jobId":{id}}}"#),
    );
    let result = &node["result"];
    // The wire names, pinned: camelCase, present, and shaped like `strategy`.
    assert_eq!(
        result["evDetail"].as_array().unwrap().len(),
        result["actions"].as_array().unwrap().len()
    );
    assert_eq!(result["isLocked"], false);
}
