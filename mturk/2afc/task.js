SERVER_TASKS_URL = "https://mlcloud2.rzimmermann.com/mturk-data/experiments/";

function initialize() {
  const debug = new URL(window.location.href).searchParams.get("debug");
  const debug_mode = debug !== null;
  if (debug_mode) {
    runExperiment();
  } else {
    if (window.opener == null) {
      alert("Please start tasks only via the Start page.");
      window.location = "start.html";
    } else {
      // check whether the user has access to the task
      runExperiment();
    }
  }
}

function runExperiment(type) {
  // get our internal ID to load the correct images
  let url = new URL(window.location.href);
  let taskId = url.searchParams.get("tid");
  let taskNamespace = url.searchParams.get("tns");
  let experimentName = url.searchParams.get("exp");

  let noInstructions = url.searchParams.get("ni");
  noInstructions = noInstructions !== null;
  
  let noDemo = url.searchParams.get("nd");
  noDemo = noDemo !== null;

  let taskIndexUrl = new URL(
    `${experimentName}/${taskNamespace}/task_${taskId}/index.json`,
    SERVER_TASKS_URL
  );

  let callback = prepareExperiment;
  let demoTaskIndexUrl = new URL(
    `${experimentName}/demo_${taskNamespace}/index.json`,
    SERVER_TASKS_URL
  );

  fetchAllJson(taskIndexUrl, demoTaskIndexUrl)
    .catch((e) => {
      alert("There was an error starting this HIT, please try it again.");
      // alert('You can no longer access this experiment as it has expired. Reminder: after accepting a task you must complete it within 5 minutes.')
    })
    .then((tasks) =>
      callback(...tasks, taskId, experimentName, taskNamespace, noInstructions, noDemo)
    );
}

function prepareExperiment(
  main_task_config,
  demo_task_config,
  taskId,
  experimentName,
  taskNamespace,
  noInstructions,
  noDemo
) {
  let timeline = [];

  // set random generator
  Math.seedrandom(main_task_config["task_name"]);

  const correct_trials_counter = {
    count: 0
  };

  function addTrials(trials, timeline, is_demo, start_progress, end_progress) {
    let task_timeline = [
      {
        type: "2afc-image-confidence-response",
        query_a_stimulus: jsPsych.timelineVariable("min_query"),
        query_b_stimulus: jsPsych.timelineVariable("max_query"),
        reference_a_stimuli: jsPsych.timelineVariable("min_references"),
        reference_b_stimuli: jsPsych.timelineVariable("max_references"),
        choices: ["1", "2", "3", "1", "2", "3"],
        prompt:
          "<p>Which image at the center matches the <b>Right Examples</b> better?</p>",
        correct_text: "",
        reference_a_title: "Left Examples",
        reference_b_title: "Right Examples",
        incorrect_text: "",
        feedback_delay_duration: 350,
        feedback_duration: 1150,
        initial_wait_duration: 1500,
        randomize_queries: true,
        response_ends_trial: false,
        correct_query_choice: "b",
        trial_completed_callback: is_demo ? undefined : jsPsych.timelineVariable(
            "trial_completed_callback"),
        on_presentation_callback: is_demo ? undefined : jsPsych.timelineVariable(
            "on_presentation_callback"),
        data: {
          id: jsPsych.timelineVariable("task_id"),
          min_query: jsPsych.timelineVariable("min_query"),
          max_query: jsPsych.timelineVariable("max_query"),
          min_references: jsPsych.timelineVariable("min_references"),
          max_references: jsPsych.timelineVariable("max_references"),
          catch_trial: jsPsych.timelineVariable("catch_trial"),
          is_demo: is_demo,
        },
        on_finish: function () {
          jsPsych.setProgressBar(jsPsych.timelineVariable("progress")());
        },
      },
    ];

    timeline.push({
      timeline: task_timeline,
      timeline_variables: trials.map(function (trial, i) {
        return {
          trial_id: trial.id,
          progress:
            start_progress +
            ((end_progress - start_progress) / trials.length) * (i + 1),
          min_query: trial.min_query,
          max_query: trial.max_query,
          min_references: [trial.min_references[0]].concat(jsPsych.randomization.shuffle(trial.min_references.slice(1,9))),
          max_references: [trial.max_references[0]].concat(jsPsych.randomization.shuffle(trial.max_references.slice(1,9))),
          catch_trial: trial.catch_trial,
          on_presentation_callback: (display_element) => display_element.innerHTML = `<p style="text-align: left">Correct Answers: <b>${correct_trials_counter.count}</b></p>` + display_element.innerHTML,
          trial_completed_callback: (trial_response) => correct_trials_counter.count += trial_response.correct ? 1 : 0,
        };
      }),
    });
  }

  let instructionImages = [];
  if (!noInstructions) {
    let welcome = {
      type: "instructions",
      pages: ["Welcome to this experiment! </br> It will start soon."],
      show_clickable_nav: true,
      on_finish: function () {
        jsPsych.setProgressBar(0.01);
      },
    };
    timeline.push(welcome);

    instructionImages = Array.from({ length: 13 }, (_, i) => i + 1).map(
      (i) =>
        new URL(
          `${experimentName}/${
            main_task_config["task_name"].split("/")[0]
          }/instructions/${i}.jpg`,
          SERVER_TASKS_URL
        )
    );

    const synthetic_instructions = [
      "In this experiment, you will be shown images on the screen and <br> asked to make a response by clicking your mouse.",
      "The experiment consists of multiple responses like this. <br> We will now explain to you how a single trial works.",
      `<br><br>On the left and the right side of the screen, you see two groups of example images. <br> They are labeled <b>Left Examples</b> and <b>Right Examples</b>.  <br> <img id="jspsych-instructions-image" src="${instructionImages[0]}" />`,
      `<br><br>At the center of the screen, you see two more images. <br> While one image belongs to the Left Examples, the other one belongs to the Right Examples. <br> <img id="jspsych-instructions-image" src="${instructionImages[1]}" />`,
      `<br><br>The question you have to answer is always the following: <br> Which image at the center matches the <b>Right Examples</b> better? <br> <img id="jspsych-instructions-image" src="${instructionImages[2]}" />`,
      `<br><br>Here is how you answer:<br>Below the upper center image you see a row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[3]}" />`,
      `<br><br><br> Above the lower center image you also see a row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[4]}" />`,
      `<br><br> If you think the <b>upper</b> image better matches the Right Examples, <br> choose a number from the <b>upper</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[5]}" />`,
      `<br><br> If you think the <b>lower</b> image better matches the Right Examples, <br> choose a number from the <b>lower</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[6]}" />`,
      `<br>The value of the number indicates how confident you are in your choice: <br> The higher the number, the higher your confidence. <br> If you are not sure, go with your best guess.  <br> <img id="jspsych-instructions-image" src="${instructionImages[7]}" />`,
      `<br><br>The first image of each group of example images is special and might look slightly different from the rest of the group. <br> When in doubt, pay more attention to this image. <br> <img id="jspsych-instructions-image" src="${instructionImages[12]}" />`,
      `<br><br><br>Once you provided your answer, a black frame appears around your chosen image.  <br> <img id="jspsych-instructions-image" src="${instructionImages[8]}" />`,
      `<br><br>Finally, you receive feedback: <br>If you chose the image that truly belongs to the Right Examples a green frame will appear; otherwise, it will be red.  <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
      `<br>This is the end of one trial. <br> Please note that each trial is independent of all other trials. <br> This means that you cannot transfer from one to another trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
      `<br><br><br>By clicking on the <it>Continue</it> button you continue to the next trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[10]}" />`, //TODO: change screenshot
      `<br>This is the last opportunity to go back and re-read the instructions via the <it>Previous</it> button. <br>Otherwise, we will start with a couple demo trials <br>so that you can familiarize yourself with the experiment. <br> <img id="jspsych-instructions-image" src="${instructionImages[11]}" />`,
    ];

    const natural_instructions = [
      "In this experiment, you will be shown images on the screen and <br> asked to make a response by clicking your mouse.",
      "The experiment consists of multiple responses like this. <br> We will now explain to you how a single trial works.",
      `<br><br>On the left and the right side of the screen, you see two groups of example images. <br> They are labeled <b>Left Examples</b> and <b>Right Examples</b>.  <br> <img id="jspsych-instructions-image" src="${instructionImages[0]}" />`,
      `<br><br>At the center of the screen, you see two more images. <br> While one image belongs to the Left Examples, the other one belongs to the Right Examples. <br> <img id="jspsych-instructions-image" src="${instructionImages[1]}" />`,
      `<br><br>The question you have to answer is always the following: <br> Which image at the center matches the <b>Right Examples</b> better? <br> <img id="jspsych-instructions-image" src="${instructionImages[2]}" />`,
      `<br><br>Here is how you answer:<br>Below the upper center image you see a row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[3]}" />`,
      `<br><br><br> Above the lower center image you also see a row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[4]}" />`,
      `<br><br> If you think the <b>upper</b> image better matches the Right Examples, <br> choose a number from the <b>upper</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[5]}" />`,
      `<br><br> If you think the <b>lower</b> image better matches the Right Examples, <br> choose a number from the <b>lower</b> row of numbers. <br> <img id="jspsych-instructions-image" src="${instructionImages[6]}" />`,
      `<br>The value of the number indicates how confident you are in your choice: <br> The higher the number, the higher your confidence. <br> If you are not sure, go with your best guess.  <br> <img id="jspsych-instructions-image" src="${instructionImages[7]}" />`,
      `<br><br><br>Once you provided your answer, a black frame appears around your chosen image.  <br> <img id="jspsych-instructions-image" src="${instructionImages[8]}" />`,
      `<br><br>Finally, you receive feedback: <br>If you chose the image that truly belongs to the Right Examples a green frame will appear; otherwise, it will be red.  <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
      `<br>This is the end of one trial. <br> Please note that each trial is independent of all other trials. <br> This means that you cannot transfer from one to another trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[9]}" />`,
      `<br><br><br>By clicking on the <it>Continue</it> button you continue to the next trial. <br> <img id="jspsych-instructions-image" src="${instructionImages[10]}" />`, //TODO: change screenshot
      `<br>This is the last opportunity to go back and re-read the instructions via the <it>Previous</it> button. <br>Otherwise, we will start with a couple demo trials <br>so that you can familiarize yourself with the experiment. <br> <img id="jspsych-instructions-image" src="${instructionImages[11]}" />`,
    ]

    const is_natural = taskNamespace.includes("natural");

    if (is_natural) {
      // Remove last instruction image as this only applies to synthetic condition.
      instructionImages = instructionImages.slice(0, -1);
    }

    let instructions = {
      timeline: [
        {
          type: "instructions",
          pages: is_natural ? natural_instructions : synthetic_instructions,
          images: [null, null].concat(instructionImages),
          show_clickable_nav: true,
          on_finish: function () {
            jsPsych.setProgressBar(0.05);
          },
        },
      ],
    };
    timeline.push(instructions);
  }

  timeline.push({
    type: "fullscreen",
    fullscreen_mode: true,
    message:
      "<p>The experiment will switch to full screen mode when you press the button below.</p>",
  });

  function getTaskStructure(task_config) {
    const taskName = task_config["task_name"];
    const nTrials = task_config["n_trials"];
    const nReferenceImages = task_config["n_reference_images"];

    let catchTrialIdxs;
    if ("catch_trial_idxs" in task_config) {
      catchTrialIdxs = task_config["catch_trial_idxs"];
    } else {
      catchTrialIdxs = [];
    }

    let maxReferenceImageIdsPerTrial = [];
    let minReferenceImageIdsPerTrial = [];
    // NOTE: images used to be numbered 0-9, with min_0 being the min-query and max_9 being the max-query.
    // for (let i = 8; i > 8 - nReferenceImages; i--) {
    //   maxReferenceImageIdsPerTrial.push("max_" + i + ".png");
    //   minReferenceImageIdsPerTrial.push("min_" + (i + 1) + ".png");
    // }
    for (let i = 8; i > 8 - nReferenceImages; i--) {
      maxReferenceImageIdsPerTrial.push("max_" + i + ".png");
      minReferenceImageIdsPerTrial.push("min_" + i + ".png");
    }

    let trialIds = Array.from(Array(nTrials).keys()).map((x) => x + 1);

    let taskStructure = {
      trials: trialIds.map((trialId) => ({
        max_references: maxReferenceImageIdsPerTrial.map(
          (imageId) =>
            new URL(
              `${experimentName}/${taskName}/trials/trial_${trialId}/references/${imageId}`,
              SERVER_TASKS_URL
            )
        ),
        min_references: minReferenceImageIdsPerTrial.map(
          (imageId) =>
            new URL(
              `${experimentName}/${taskName}/trials/trial_${trialId}/references/${imageId}`,
              SERVER_TASKS_URL
            )
        ),
        max_query: new URL(
          `${experimentName}/${taskName}/trials/trial_${trialId}/queries/max.png`,
          SERVER_TASKS_URL
        ),
        min_query: new URL(
          `${experimentName}/${taskName}/trials/trial_${trialId}/queries/min.png`,
          SERVER_TASKS_URL
        ),
        id: trialId,
        catch_trial: catchTrialIdxs.includes(trialId),
      })),
      length: nTrials,
    };

    return taskStructure;
  }

  const main_task_structure = getTaskStructure(main_task_config);
  const demo_task_structure = getTaskStructure(demo_task_config);

  const main_task_trials = [].concat.apply([], main_task_structure["trials"]);
  const demo_task_trials = [].concat.apply([], demo_task_structure["trials"]);

  const main_task_images = [].concat.apply(
    [],
    main_task_trials.map((trial) =>
      [trial["max_query"], trial["min_query"]].concat(
        trial["max_references"],
        trial["min_references"]
      )
    )
  );
  const demo_task_images = [].concat.apply(
    [],
    demo_task_trials.map((trial) =>
      [trial["max_query"], trial["min_query"]].concat(
        trial["max_references"],
        trial["min_references"]
      )
    )
  );
  const images = main_task_images.concat(demo_task_images);

  if (!noDemo) {
    let demo_trials_timeline = [];
    addTrials(
      demo_task_structure["trials"],
      demo_trials_timeline,
      true,
      0.1,
      0.2
    );

    // the obvious trials refers to the trials that were hand-picked to be very easy
    // while the non-obvious trials were also hand-picked to be easy but are not as
    // easy as the other ones
    const n_obvious_demo_trials = demo_task_config["n_obvious_trials"];
    const obvious_demo_trials_variables =
      demo_trials_timeline[0].timeline_variables.slice(
        0,
        n_obvious_demo_trials
      );
    const non_obvious_demo_trials_variables =
      demo_trials_timeline[0].timeline_variables.slice(n_obvious_demo_trials);

    const n_obvious_trials_to_show = 4;
    const n_non_obvious_trials_to_show = 1;

    // create every possible permutation of the practice trials
    // but permute only the obvious and the non-obvious trials within their groups
    // but not across these
    let possible_demo_trials_timelines = getRandomSubset(
        permutator(obvious_demo_trials_variables), 10).map((obvs_vars) =>
            getRandomSubset(permutator(non_obvious_demo_trials_variables), 10).map(
                (non_obvs_vars) => [{
                  timeline: demo_trials_timeline[0].timeline,
                timeline_variables: jsPsych.randomization.shuffle(
                    obvs_vars
                        .slice(0, n_obvious_trials_to_show)
                        .concat(non_obvs_vars.slice(0, n_non_obvious_trials_to_show))
                ).map((it, idx) => ({
                    ...it,
                    progress:
                      0.1 +
                      ((0.2 - 0.1) /
                        (n_obvious_trials_to_show + n_non_obvious_trials_to_show)) *
                        (idx + 1),
                  })),
              },
            ])
          )
          .flat();

    function createSwitchTimeline(items) {
      return [
        // create switch head that draws a random number
        {
          type: "call-function",
          func: () => getRandomInt(0, items.length - 1),
        },
      ].concat(
        // now test for each item whether the random value matches their index
        items.map((item, idx) => {
          return {
            timeline: item,
            conditional_function: () => {
              const data = jsPsych.data.getLastTimelineData();
              const random_value = data.values().last().value;
              return idx == random_value;
            },
          };
        })
      );
    }

    let demo_timeline = [
      {
        timeline: createSwitchTimeline(possible_demo_trials_timelines).concat([
          {
            timeline: [
              {
                type: "instructions",
                pages: [
                  "As you did not answer all trials correctly, we'd like you to repeat them.",
                ],
                show_clickable_nav: true,
                allow_backward: false,
                on_finish: function () {
                  jsPsych.setProgressBar(0.1);
                },
              },
            ],
            conditional_function: function () {
              let data = jsPsych.data.getLastTimelineData();
              return !data
                .filter({ trial_type: "2afc-image-confidence-response" })
                .values()
                .every((i) => i.correct);
            },
          },
          {
            timeline: [
              {
                type: "instructions",
                pages: ["Great! Let's now start with the real trials!"],
                show_clickable_nav: true,
                allow_backward: false,
                key_forward: "Enter",
                button_label_next: "Continue (Enter)",
              },
            ],
            conditional_function: function () {
              let data = jsPsych.data.getLastTimelineData();
              // if the previous trial was the conditional instruction then data
              // will contain only this item
              return (
                data.count() > 1 &&
                data
                  .filter({ trial_type: "2afc-image-confidence-response" })
                  .values()
                  .every((i) => i.correct)
              );
            },
            show_clickable_nav: true,
          },
        ]),
        loop_function: function (data) {
          return !data
            .filter({ trial_type: "2afc-image-confidence-response" })
            .values()
            .every((i) => i.correct);
        },
      },
    ];
    timeline = timeline.concat(demo_timeline);
  }

  // The progress bar will end at 0.95 with the trials. This is supposed to prevent people from
  // closing the tab before the data is transferred.
  addTrials(main_task_structure["trials"], timeline, false, 0.2, 0.95);

  let feedback_trial = {
    type: "survey-text",
    questions: [
      {
        prompt: "Optional: Is there any feedback you'd like to share with us?",
        name: "feedback",
        rows: 8,
        columns: 80,
        required: false,
      },
    ],
  };
  timeline.push(feedback_trial);

  let sending_warning = {
    type: "instructions",
    pages: [
      "Press Continue to submit your data; this can take a moment. <br> Please do not close the window until we tell you to do so.",
    ],
    show_clickable_nav: true,
    allow_backward: false,
    key_forward: "Enter",
    button_label_next: "Continue (Enter)",
  };
  timeline.push(sending_warning);

  let send_response_payload = {
    type: "call-function",
    async: true,
    func: function (callback) {
      function sendMTurkPayload() {
        // send results back to MTurk
        const rawData = jsPsych.data.getAllData();
        const mainData = jsPsych.data
          .getLastTimelineData()
          .filter({ trial_type: "2afc-image-confidence-response" });
        const rawPayload = rawData.json();
        const mainPayload = mainData.json();
        const json_data = JSON.stringify({
          main_data: mainPayload,
          raw_data: rawPayload,
          task_id: taskId,
        });

        window.opener.postMessage(json_data, window.opener.location.href);
      }

      let url = new URL(window.location.href);
      const noBouncerFlag = url.searchParams.get("nb");
      const noBouncer = noBouncerFlag !== null;

      if (noBouncer) {
        sendMTurkPayload();
        callback();
      } else {
        // send request to bouncer to make sure the worker cannot participate again
        let bouncer_url = new URL(
          "https://mlcloud2.rzimmermann.com/mturk/bouncer/ban"
        );
        let url = new URL(window.location.href);
        let turk_info = jsPsych.turk.turkInfo();
        bouncer_url.searchParams.append("wid", turk_info.workerId);
        bouncer_url.searchParams.append("eid", url.searchParams.get("exp"));
        bouncer_url.searchParams.append("tns", url.searchParams.get("tns"));
        fetchJson(bouncer_url).finally(() => {
          sendMTurkPayload();
          callback();
        });
      }
    },
    on_finish: function () {
      jsPsych.setProgressBar(0.95);
    },
  };
  timeline.push(send_response_payload);

  let end = {
    type: "html-keyboard-response",
    stimulus:
      '<p style="color: white;">Your responses have been saved and submitted. </br></br>Thanks for your participation!</br></br>This window will automatically be closed in 5 seconds. Feel free to close it now already.</p>',
    choices: jsPsych.NO_KEYS,
    on_start: function (trial) {
      jsPsych.pluginAPI.setTimeout(function () {
        window.close();
      }, 5000);
    },
    on_finish: function () {
      jsPsych.setProgressBar(1.0);
    },
  };
  timeline.push(end);

  jsPsych.init({
    timeline: timeline,
    exclusions: {
      min_width: 940,
      min_height: 600,
    },
    preload_images: images.concat(instructionImages),
    show_preload_progress_bar: true,
    show_progress_bar: true,
    auto_update_progress_bar: false,
  });
}
