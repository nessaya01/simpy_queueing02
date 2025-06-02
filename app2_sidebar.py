# app2_sidebar.py — Streamlit Simulation Dashboard

import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches

from scipy import stats
from xlsxwriter import Workbook

from simulation_engine import run_single_replication, run_simulation_multi

#---------------------------------------
# APP LAYOUT: PAGE
#---------------------------------------

st.set_page_config(page_title="Simulation Dashboard", page_icon=":bar_chart:", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to section:",
    ("Welcome", "Simulation Setup", "Results & Plots", "Logs & Exports", "Help & Guide")
)


st.sidebar.markdown("---")
st.sidebar.info("Author: Luis Herrera\nVersion: 2.0")
st.sidebar.image("LuisHerrera.jpg", width=160, caption="Luis Herrera")

#---------------------------------------
# PAGE: WELCOME
#---------------------------------------
if page == "Welcome":
    st.header("Welcome to the Simulation Dashboard")
    st.markdown("""
    This app lets you simulate, analyze, and visualize complex queueing systems with a wide range of configurable features:

- **Multiple Entity Types:** Customize name, priority, service times, and arrival probability for each entity class.
- **Time-Varying Arrival Rates:** Model distinct arrival intensities for morning, peak, and late periods.
- **Batch Arrivals:** Simulate groups of entities arriving together.
- **Multiple Servers:** Configure any number of parallel servers.
- **Server Breakdowns and Repairs:** Randomly take servers down and repair them, with full downtime tracking.
- **Queue Prioritization & Aging:** Assign and dynamically adjust priorities, including aging logic.
- **Reneging:** Entities may leave the queue if they wait too long.
- **Replications & Confidence Intervals:** Run multiple experiments and get robust summary statistics.
- **Queue & System Monitoring:** Track average queue length and system congestion.
- **Server Utilization:** Analyze both scheduled and real (breakdown-adjusted) utilization.
- **Comprehensive Event Log:** Every arrival, service, reneging, breakdown/repair, and more is logged and exportable.
- **Entity-Level Gantt Timeline:** Visualize the flow of each entity through the system.
- **Multiple Visualizations:** Arrival histograms, type-based bar charts, utilization plots, and more.
- **Full CSV/Excel Export:** Export all logs, metrics, and event data.
    """)
    
    st.info("Use the sidebar to navigate the app. Start with Simulation Setup.")


#---------------------------------------
# PAGE: SIMULATION SETUP — SIDEBAR PAGE (harmonized version)
#---------------------------------------

elif page == "Simulation Setup":

    st.header("Simulation Setup")
    
    # --- Expand/Collapse All Logic ---
    if "expand_all" not in st.session_state:
        st.session_state["expand_all"] = True
    expand_label = "Collapse All Sections" if st.session_state["expand_all"] else "Expand All Sections"
    if st.button(expand_label, key="expand_all_btn"):
        st.session_state["expand_all"] = not st.session_state["expand_all"]

    # --- Visual Explanation GIF ---
    with st.expander("How This Simulation Works (Visual)", expanded=False):
        st.image(
            "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExNWI4cXV3eXBjOXl4NXhvdHFsZzZyZGQyMmxmcGhrZTQ2M2hwaGUzeiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT5LMuHy92KbOfnd8A/giphy.gif",
            caption="Entities arrive, wait in a queue, are served, and depart — just like in this simulation!",
            use_column_width=True,
        )
        st.markdown("""
        In this simulation, entities (people, jobs, products) arrive according to a schedule, may have to wait in a queue, are served by one or more servers, and then depart the system. You can model all these dynamics — including priority, breakdowns, batching, reneging, and more!
        """)
    
    # --- Quick Start Presets ---
    with st.expander("Quick Start Presets (Example Scenarios)", expanded=False):
        st.write("Instantly load parameters for typical queueing scenarios.")
    
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Small System", key="preset_small"):
                st.session_state['entity_types'] = [
                    {"name": "Type1", "priority": 0, "service_time": 5.0, "prob": 1.0}
                ]
                st.session_state['sim_time'] = 480
                st.session_state['num_servers'] = 3
                st.session_state['replications'] = 6
                st.session_state['enable_batch'] = False
                st.session_state['failure_enabled'] = False
                st.rerun()
        with col2:
            if st.button("Heavy Load", key="preset_heavy"):
                st.session_state['entity_types'] = [
                    {"name": "Type1", "priority": 0, "service_time": 6.0, "prob": 0.5},
                    {"name": "Type2", "priority": 1, "service_time": 9.0, "prob": 0.5}
                ]
                st.session_state['sim_time'] = 480
                st.session_state['num_servers'] = 1
                st.session_state['replications'] = 6
                st.session_state['enable_batch'] = True
                st.session_state['batch_min'] = 2
                st.session_state['batch_max'] = 5
                st.session_state['failure_enabled'] = True
                st.session_state['mtbf'] = 60.0
                st.session_state['mttr'] = 15.0
                st.rerun()
        with col3:
            if st.button("Balanced Mix", key="preset_balanced"):
                st.session_state['entity_types'] = [
                    {"name": "TypeA", "priority": 0, "service_time": 7.0, "prob": 0.4},
                    {"name": "TypeB", "priority": 1, "service_time": 12.0, "prob": 0.6}
                ]
                st.session_state['sim_time'] = 480
                st.session_state['num_servers'] = 3
                st.session_state['replications'] = 6
                st.session_state['enable_batch'] = False
                st.session_state['failure_enabled'] = False
                st.rerun()
        st.caption("Select a preset to fill all simulation parameters instantly. You can still adjust any values afterwards.")
    
    st.markdown(
        "<hr style='height:5px; border:none; background: linear-gradient(to right, #343a40, #dee2e6, #f8f9fa, #dee2e6, #343a40);'/>",
        unsafe_allow_html=True
    )
    
    # --- GENERAL SIMULATION SETTINGS ---
    with st.expander("General Simulation Settings", expanded=st.session_state["expand_all"]):
        sim_time = st.number_input(
            "Total Simulation Time (minutes)", min_value=1, max_value=10000, value=100,
            help="How long to run each simulation replication."
        )
        warmup_time = st.number_input(
            "Warm-up Period (minutes)", min_value=0, max_value=int(sim_time), value=10,
            help="Initial period (minutes) to ignore in statistics (system stabilization)."
        )
        replications = st.number_input(
            "Number of Replications", min_value=1, max_value=100, value=5,
            help="How many times to run the simulation with the same parameters (for confidence intervals)."
        )
        num_servers = st.number_input(
            "Number of Servers", min_value=1, max_value=100, value=2,
            help="Number of parallel servers in the system."
        )
        queue_monitor_interval = st.number_input(
            "Queue Monitoring Interval (minutes)", min_value=0.01, max_value=10.0, value=0.5,
            help="How frequently to sample the queue for average length."
        )
    
    # --- ENTITY TYPES ---
    with st.expander("Entity Types (Add/Remove/Edit)", expanded=st.session_state["expand_all"]):
        st.write("Define the different classes of entities for your simulation.")
        if 'entity_types' not in st.session_state:
            st.session_state.entity_types = [
                {"name": "Type1", "priority": 0, "service_time": 6.0, "prob": 0.5},
                {"name": "Type2", "priority": 1, "service_time": 9.0, "prob": 0.5}
            ]
        entity_types = st.session_state.entity_types
        for idx, ent in enumerate(entity_types):
            cols = st.columns([2, 1, 2, 2, 1])
            with cols[0]:
                ent["name"] = st.text_input(f"Type name {idx+1}", value=ent["name"], key=f"name_{idx}")
            with cols[1]:
                ent["priority"] = st.number_input(
                    f"Priority (0 = highest) {idx+1}", min_value=0, max_value=20,
                    value=int(ent["priority"]), key=f"priority_{idx}")
            with cols[2]:
                ent["service_time"] = st.number_input(
                    f"Avg Service Time (min) {idx+1}", min_value=0.01, max_value=100.0,
                    value=float(ent["service_time"]), key=f"service_{idx}")
            with cols[3]:
                ent["prob"] = st.number_input(
                    f"Arrival Probability {idx+1}", min_value=0.0, max_value=1.0,
                    value=float(ent["prob"]), key=f"prob_{idx}")
            with cols[4]:
                if st.button("Remove", key=f"remove_{idx}"):
                    entity_types.pop(idx)
                    st.experimental_rerun()
        if st.button("Add Entity Type"):
            entity_types.append(
                {"name": f"Type{len(entity_types)+1}", "priority": len(entity_types),
                 "service_time": 6.0, "prob": 0.1}
            )
        # Normalize probabilities
        total_prob = sum(ent["prob"] for ent in entity_types)
        if total_prob > 0:
            for ent in entity_types:
                ent["prob"] = round(ent["prob"] / total_prob, 3)
        st.caption("Arrival probabilities will be automatically normalized to sum to 1.")
    
    # --- ARRIVAL PATTERN ---
    with st.expander("Arrival Pattern (Time-Varying Interarrival Rates)", expanded=st.session_state["expand_all"]):
        st.write("Specify the average interarrival time for different periods.")
        split_even = st.radio(
            "How do you want to split the simulation time into periods?",
            options=["Evenly (1/3, 1/3, 1/3)", "Custom (user-defined durations)"],
            index=0,
            help="Choose 'Evenly' to split total time into three equal periods, or 'Custom' to set durations for each."
        )
        if split_even == "Evenly (1/3, 1/3, 1/3)":
            period1 = sim_time / 3
            period2 = sim_time / 3
            period3 = sim_time - period1 - period2
        else:
            period1 = st.number_input("Duration of Morning Period (minutes)", min_value=1, max_value=int(sim_time), value=int(sim_time/3))
            period2 = st.number_input("Duration of Peak Period (minutes)", min_value=1, max_value=int(sim_time - period1), value=int(sim_time/3))
            period3 = sim_time - period1 - period2
            st.write(f"Duration of Late Period (auto-calculated): **{period3}** minutes")
            if period3 < 1:
                st.error("The sum of morning and peak durations cannot exceed total simulation time.")
    
        morning_rate = st.number_input(
            f"Avg inter-arrival time during Morning period (0–{int(period1)} min)",
            min_value=0.01, max_value=100.0, value=5.0
        )
        peak_rate = st.number_input(
            f"Avg inter-arrival time during Peak period ({int(period1)}–{int(period1+period2)} min)",
            min_value=0.01, max_value=100.0, value=3.0
        )
        late_rate = st.number_input(
            f"Avg inter-arrival time during Late period ({int(period1+period2)}–{int(sim_time)} min)",
            min_value=0.01, max_value=100.0, value=5.0
        )
    
    # --- BATCH ARRIVALS ---
    with st.expander("Batch Arrivals", expanded=st.session_state["expand_all"]):
        st.write("Enable and configure batch arrivals if you want multiple entities to arrive together.")
        enable_batch = st.checkbox("Enable batch arrivals?", value=False)
        if enable_batch:
            batch_min = st.number_input("Minimum batch size", min_value=1, max_value=100, value=1)
            batch_max = st.number_input("Maximum batch size", min_value=batch_min, max_value=100, value=3)
            st.caption("For each arrival event, a random number of entities between minimum and maximum batch size will be created.")
        else:
            batch_min = batch_max = 1
    
    # --- SERVER BREAKDOWNS ---
    with st.expander("Server Breakdowns (Failures & Repairs)", expanded=st.session_state["expand_all"]):
        st.write("Enable random breakdowns and repairs for each server.")
        failure_enabled = st.checkbox("Enable server breakdowns?", value=False)
        if failure_enabled:
            mtbf = st.number_input(
                "Mean Time Between Failures (MTBF, minutes)", min_value=0.01, max_value=1e5, value=120.0,
                help="Average time until a server fails."
            )
            mttr = st.number_input(
                "Mean Time To Repair (MTTR, minutes)", min_value=0.01, max_value=1e5, value=20.0,
                help="Average time needed to repair a failed server."
            )
            st.caption("Each server will operate for a random duration (on average MTBF), then break down and be unavailable for a random repair period (on average MTTR).")
        else:
            mtbf = None
            mttr = None
    
    # --- PRIORITIES & AGING ---
    with st.expander("Priorities & Aging", expanded=st.session_state["expand_all"]):
        st.write("Configure dynamic adjustment (aging) of entity priorities while waiting.")
        aging_interval = st.number_input(
            "Time between priority improvements (minutes)", min_value=0.01, max_value=1e3, value=5.0,
            help="How long an entity waits before its priority is improved (aging)."
        )
        max_aging_steps = st.number_input(
            "Maximum number of priority improvements (aging steps)", min_value=1, max_value=20, value=2,
            help="Maximum number of times an entity's priority can be improved."
        )
        min_priority = st.number_input(
            "Minimum priority (0 = highest)", min_value=0, max_value=100, value=0,
            help="Lowest allowed priority value; cannot improve beyond this."
        )
    
    # --- RENEGING ---
    with st.expander("Reneging (Impatient Entities)", expanded=st.session_state["expand_all"]):
        st.write("Entities may leave the queue if they wait too long without service.")
        renege_time = st.number_input(
            "Max waiting time before reneging (minutes)", min_value=0.01, max_value=1e5, value=12.0,
            help="If an entity waits longer than this, it will leave (renege) the queue."
        )
    
    st.markdown(
        "<hr style='height:5px; border:none; background: linear-gradient(to right, #343a40, #dee2e6, #f8f9fa, #dee2e6, #343a40);'/>",
        unsafe_allow_html=True
    )
    
    # --- PARAMETER SUMMARY PREVIEW ---
    with st.expander("Parameter Summary Preview", expanded=False):
        st.markdown("### Parameter Summary Preview")
        param_preview = {
            "Simulation Time (min)": sim_time,
            "Warm-up Period (min)": warmup_time,
            "Replications": replications,
            "Servers": num_servers,
            "Queue Monitor Interval": queue_monitor_interval,
            "Arrival Pattern (min)": f"Morning: {period1}, Peak: {period2}, Late: {period3}",
            "Inter-arrival (min)": f"Morning: {morning_rate}, Peak: {peak_rate}, Late: {late_rate}",
            "Batch Arrivals": "Enabled" if enable_batch else "Disabled",
            "Batch Size": f"{batch_min}–{batch_max}" if enable_batch else "1",
            "Server Breakdowns": "Enabled" if failure_enabled else "Disabled",
            "MTBF": mtbf if failure_enabled else "-",
            "MTTR": mttr if failure_enabled else "-",
            "Priority Aging (interval)": aging_interval,
            "Max Aging Steps": max_aging_steps,
            "Min Priority": min_priority,
            "Reneging Time": renege_time
        }
        st.table(pd.DataFrame(param_preview.items(), columns=["Parameter", "Value"]))
        # Entity Types Summary Table
        et_df = pd.DataFrame(st.session_state.entity_types)
        st.markdown("**Entity Types Defined:**")
        st.table(et_df.rename(columns={
            "name": "Type",
            "priority": "Priority",
            "service_time": "Avg Service Time",
            "prob": "Arrival Probability"
        }))
    
    # --- RUN SIMULATION BUTTON ---
    run_sim = st.button("Run Simulation", type="primary")
    if run_sim:
        # Save the config
        config = {
            "SIM_TIME": sim_time,
            "WARMUP_TIME": warmup_time,
            "REPLICATIONS": replications,
            "NUM_SERVERS": num_servers,
            "QUEUE_MONITOR_INTERVAL": queue_monitor_interval,
            "ENTITY_TYPES": st.session_state.entity_types.copy(),
            "PERIOD1": period1,
            "PERIOD2": period2,
            "PERIOD3": period3,
            "MORNING_RATE": morning_rate,
            "PEAK_RATE": peak_rate,
            "LATE_RATE": late_rate,
            "ENABLE_BATCH": enable_batch,
            "BATCH_MIN": batch_min,
            "BATCH_MAX": batch_max,
            "FAILURE_ENABLED": failure_enabled,
            "MTBF": mtbf,
            "MTTR": mttr,
            "AGING_INTERVAL": aging_interval,
            "MAX_AGING_STEPS": max_aging_steps,
            "MIN_PRIORITY": min_priority,
            "RENEGE_TIME": renege_time
        }
        st.session_state.run_config = config
    
        # --- THIS IS WHAT YOU ARE MISSING ---
        # Actually run the simulation and save results
        (
            all_summaries,
            all_entity_logs,
            all_event_logs,
            all_queue_times,
            all_queue_lengths,
            all_utilizations
        ) = run_simulation_multi(config)
        st.session_state.sim_results = (
            all_summaries,
            all_entity_logs,
            all_event_logs,
            all_queue_times,
            all_queue_lengths,
            all_utilizations
        )
        st.success("Simulation complete! Please go to the 'Results & Plots' tab to view outputs.")
        st.write("✅ Simulation run? sim_results in session_state:", "sim_results" in st.session_state)

# ================== END SIMULATION SETUP SIDEBAR PAGE ==================

#---------------------------------------
# PAGE: RESULTS AND PLOTS
#---------------------------------------

elif page == "Results & Plots":

    st.header("Results & Plots")

    if "sim_results" not in st.session_state:
        st.info("Please run the simulation from the setup page first.")
    else:
        (
            all_summaries,
            all_entity_logs,
            all_event_logs,
            all_queue_times,
            all_queue_lengths,
            all_utilizations
        ) = st.session_state.sim_results

        st.subheader("Replication Summaries")
        df_reps = pd.DataFrame(all_summaries)
        st.dataframe(df_reps)

        # --------------------------
        # Confidence Intervals
        # --------------------------
        def ci(data):
            mean = np.mean(data)
            sem = stats.sem(data)
            if len(data) > 1:
                ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
            else:
                ci_low, ci_high = mean, mean
            return round(mean, 4), round(ci_low, 4), round(ci_high, 4)

        cols_to_ci = [col for col in df_reps.columns if 'Avg' in col or col.endswith("_Count") or "Utilization" in col]
        summary_ci = {}
        for col in cols_to_ci:
            if df_reps[col].notna().sum() > 1:
                summary_ci[col] = dict(zip(["Mean", "CI Low", "CI High"], ci(df_reps[col])))
        if summary_ci:
            st.subheader("Confidence Intervals (95%)")
            st.dataframe(pd.DataFrame(summary_ci).T)
        else:
            st.info("Not enough data for confidence intervals.")

        # --------------------------
        # Arrival Time Histogram
        # --------------------------
        st.subheader("Histogram of All Entity Arrivals")
        df_logs_all = pd.concat(all_entity_logs, ignore_index=True)
        if "ArrivalTime" in df_logs_all.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df_logs_all["ArrivalTime"], bins=40, color="skyblue", edgecolor="black")
            ax.set_xlabel("Simulation Time")
            ax.set_ylabel("Number of Arrivals")
            ax.set_title("Histogram of All Entity Arrivals")
            ax.grid(True, axis='y', linestyle='--')
            st.pyplot(fig)
        else:
            st.warning("No arrival time data found in logs.")

        # --------------------------
        # Metrics Across Replications (utilization included)
        # --------------------------
        st.subheader("Metrics Across Replications (includes Utilization)")
        metric_cols = [col for col in df_reps.columns if col.startswith("Avg") or "_Avg" in col or col in ["AvgQueueLength", "AgedEntities", "ServerUtilization_Scheduled", "ServerUtilization_Real"]]
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in metric_cols:
            if col in df_reps:
                ax.plot(df_reps["Replication"], df_reps[col], marker='o', label=col)
        ax.set_xlabel("Replication")
        ax.set_ylabel("Value")
        ax.set_title("Metrics Across Replications (includes Utilization)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # --------------------------
        # Per Entity Type Summary and Bar Plot
        # --------------------------
        st.subheader("Per Entity Type Summary")
        entity_summary = {}
        for label in df_logs_all["Type"].unique():
            df_type = df_logs_all[df_logs_all["Type"] == label]
            df_served = df_type[~df_type["Reneged"]]
            if not df_served.empty:
                entity_summary[label] = {
                    "Avg Wait": round((df_served["StartService"] - df_served["ArrivalTime"]).mean(), 2),
                    "Avg Service": round((df_served["EndService"] - df_served["StartService"]).mean(), 2),
                    "Avg Total": round((df_served["EndService"] - df_served["ArrivalTime"]).mean(), 2),
                    "Total Served": len(df_served)
                }
        df_entity = pd.DataFrame(entity_summary).T
        st.dataframe(df_entity)

        # Bar Plot – Per Entity Type Performance
        st.subheader("Performance per Entity Type (Bar Plot)")
        if not df_entity.empty:
            x = np.arange(len(df_entity.index))
            width = 0.25
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - width, df_entity["Avg Wait"], width=width, label="Avg Wait")
            ax.bar(x, df_entity["Avg Service"], width=width, label="Avg Service")
            ax.bar(x + width, df_entity["Avg Total"], width=width, label="Avg Total")
            ax.set_xticks(x)
            ax.set_xticklabels(df_entity.index)
            ax.set_title("Performance per Entity Type")
            ax.set_xlabel("Entity Type")
            ax.set_ylabel("Time (minutes)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Not enough data for entity summary plot.")

        # --------------------------
        # Gantt-style Entity-Level Flow Timeline (with Pagination)
        # --------------------------
        st.subheader("Gantt-style Entity-Level Flow Timeline (with Pagination)")

        if not df_logs_all.empty:
            rep_options = list(df_logs_all["Replication"].unique())
            rep_options.sort()
            rep_choice = st.selectbox(
                "Choose replication to display",
                options=["All"] + rep_options,
                index=0
            )
            # Filter by replication if not "All"
            if rep_choice != "All":
                logs_to_plot = df_logs_all[df_logs_all["Replication"] == rep_choice]
            else:
                logs_to_plot = df_logs_all

            logs_to_plot = logs_to_plot.sort_values(by="ArrivalTime").reset_index(drop=True)
            total_entities = len(logs_to_plot)
            default_page_size = 50
            page_size = st.number_input("Entities per page", min_value=10, max_value=300, value=default_page_size, step=10)
            max_page = (total_entities - 1) // page_size + 1
            page_num = st.number_input("Page number", min_value=1, max_value=max_page, value=1)

            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, total_entities)
            logs_page = logs_to_plot.iloc[start_idx:end_idx]

            st.markdown(f"**Displaying entities {start_idx+1} to {end_idx} of {total_entities} (page {page_num}/{max_page})**")

            if not logs_page.empty:
                fig, ax = plt.subplots(figsize=(12, min(8, len(logs_page)//6 + 2)))
                color_map = {etype: plt.cm.tab10(i) for i, etype in enumerate(sorted(logs_page['Type'].unique()))}
                y_ticks, y_labels = [], []
                for i, row in logs_page.iterrows():
                    label = f"{row['Type']}-{row['EntityID']}"
                    y = i
                    y_ticks.append(y)
                    y_labels.append(label)
                    if row.get("Reneged", False):
                        ax.plot(row["ArrivalTime"], y, 'rx', markersize=8)
                    elif all(k in row and pd.notnull(row[k]) for k in ["StartService", "EndService"]):
                        ax.broken_barh(
                            [(row["StartService"], row["EndService"] - row["StartService"])],
                            (y - 0.4, 0.8),
                            facecolors=color_map[row["Type"]]
                        )
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels, fontsize=7)
                ax.set_xlabel("Simulation Time")
                ax.set_title(f"Entity-Level Flow Timeline (Page {page_num}/{max_page})")
                ax.grid(True)
                handles = [mpatches.Patch(color=color_map[t], label=t) for t in color_map]
                if any(logs_page["Reneged"]):
                    handles.append(plt.Line2D([], [], color='r', marker='x', linestyle='None', label='Reneged'))
                ax.legend(handles=handles, loc='upper right', fontsize=8)
                ax.set_xlim(0, st.session_state.run_config["SIM_TIME"])
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No entities to display on this page.")
        else:
            st.info("No logs to visualize.")


#---------------------------------------
# PAGE: LOGS AND EXPORTS
# You must install xlsxwriter into your current Python environment. How?
# from shell: conda install -c conda-forge xlsxwriter
# from Jupyter/Colab | python: !pip install xlsxwriter
# within Anaconda Navigator
#---------------------------------------
elif page == "Logs & Exports":
    st.header("Logs & Exports")
    
    # Check that results are available before showing logs/exports
    if "sim_results" not in st.session_state:
        st.info("Please run the simulation first.")
    else:
        (
            all_summaries,
            all_entity_logs,
            all_event_logs,
            all_queue_times,
            all_queue_lengths,
            all_utilizations
        ) = st.session_state.sim_results

        # ---- Convert lists of DataFrames to master DataFrames ----
        df_entities = pd.concat(all_entity_logs, ignore_index=True)
        df_events = pd.concat(all_event_logs, ignore_index=True)
        df_reps = pd.DataFrame(all_summaries)
        
        # ---------- A. ENTITY LOG: Interactive Preview & Filtering ----------
        st.subheader("Entity Log (All Replications)")
        # Let user filter by replication or show all
        rep_options = ["All"] + sorted([str(r) for r in df_entities["Replication"].unique()])
        rep_filter = st.selectbox("Filter by replication", rep_options, index=0)
        if rep_filter != "All":
            df_entities_view = df_entities[df_entities["Replication"] == int(rep_filter)]
        else:
            df_entities_view = df_entities
        
        # Optional: Add search by entity type
        type_options = ["All"] + sorted(df_entities_view["Type"].unique())
        type_filter = st.selectbox("Filter by entity type", type_options, index=0)
        if type_filter != "All":
            df_entities_view = df_entities_view[df_entities_view["Type"] == type_filter]
        
        ##-----Show paginated DataFrame of entity logs (Streamlit's dataframe supports search/paging)
        ##-----st.dataframe(df_entities_view)

        # Main entity log table (full view, paginated/searchable)
        st.dataframe(df_entities_view, use_container_width=True)

        
        # Preview: first/last 10 rows for rapid scan
        st.caption("First 10 rows of filtered entity log:")
        st.dataframe(df_entities_view.head(10), use_container_width=True)
        st.caption("Last 10 rows of filtered entity log:")
        st.dataframe(df_entities_view.tail(10), use_container_width=True)
        
        # Download button for entity log
        st.download_button(
            label="Download Entity Log (CSV)",
            data=df_entities_view.to_csv(index=False),
            file_name="entity_logs.csv",
            mime="text/csv"
        )

        st.markdown("---")

        # ---------- B. EVENT LOG: Interactive Preview & Filtering ----------
        st.subheader("Event Log (All Replications)")
        # Filter by replication
        rep_event_options = ["All"] + sorted([str(r) for r in df_events["Replication"].unique()])
        rep_event_filter = st.selectbox("Filter event log by replication", rep_event_options, index=0)
        if rep_event_filter != "All":
            df_events_view = df_events[df_events["Replication"] == int(rep_event_filter)]
        else:
            df_events_view = df_events

        # Show paginated DataFrame for event logs
        st.dataframe(df_events_view)
        
        # Preview head/tail
        st.caption("First 10 events:")
        st.dataframe(df_events_view.head(10))
        st.caption("Last 10 events:")
        st.dataframe(df_events_view.tail(10))

        # Download event log
        st.download_button(
            label="Download Event Log (CSV)",
            data=df_events_view.to_csv(index=False),
            file_name="event_logs.csv",
            mime="text/csv"
        )

        st.markdown("---")
        
        # ---------- C. SUMMARY AND CONFIDENCE INTERVALS EXPORT ----------
        st.subheader("Summary Table and Confidence Intervals")
        st.dataframe(df_reps)
        st.download_button(
            label="Download Replication Summaries (CSV)",
            data=df_reps.to_csv(index=False),
            file_name="replication_summaries.csv",
            mime="text/csv"
        )

        # Reuse confidence interval function from Results tab
        def ci(data):
            mean = np.mean(data)
            sem = stats.sem(data)
            if len(data) > 1:
                ci_low, ci_high = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
            else:
                ci_low, ci_high = mean, mean
            return round(mean, 4), round(ci_low, 4), round(ci_high, 4)

        cols_to_ci = [col for col in df_reps.columns if 'Avg' in col or col.endswith("_Count") or "Utilization" in col]
        summary_ci = {}
        for col in cols_to_ci:
            if df_reps[col].notna().sum() > 1:
                summary_ci[col] = dict(zip(["Mean", "CI Low", "CI High"], ci(df_reps[col])))
        if summary_ci:
            df_ci = pd.DataFrame(summary_ci).T
            st.dataframe(df_ci)
            st.download_button(
                label="Download Confidence Intervals (CSV)",
                data=df_ci.to_csv(),
                file_name="confidence_intervals.csv",
                mime="text/csv"
            )
        else:
            st.info("Not enough data for confidence intervals.")

        st.markdown("---")

        # ---------- D. OPTIONAL: Download All As Excel ----------
        st.subheader("Download All Results as Excel (Optional)")
        # Save all key tables in a single Excel file for convenience
        import io
        with io.BytesIO() as buffer:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df_entities_view.to_excel(writer, sheet_name="EntityLogs", index=False)
                df_events_view.to_excel(writer, sheet_name="EventLogs", index=False)
                df_reps.to_excel(writer, sheet_name="Replications", index=False)
                if summary_ci:
                    df_ci.to_excel(writer, sheet_name="ConfidenceIntervals", index=True)
            buffer.seek(0)
            st.download_button(
                label="Download All Results (Excel)",
                data=buffer,
                file_name="simulation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.markdown("---")

        # ---------- E. (OPTIONAL) PREVIEW OF LATEST LOGS ----------
        st.subheader("Quick Preview: Most Recent Entity and Event Logs")
        st.write("Entity (last replication):")
        if not all_entity_logs[-1].empty:
            st.dataframe(all_entity_logs[-1].tail(10))
        st.write("Event log (last replication):")
        if not all_event_logs[-1].empty:
            st.dataframe(all_event_logs[-1].tail(10))

        # More advanced filters/searches can be added as needed for your use case





#---------------------------------------
# PAGE: HELP AND GUIDE
#---------------------------------------

elif page == "Help & Guide":
    st.header("Help & Guide")
    # (Paste your help/FAQ/tips code here)


