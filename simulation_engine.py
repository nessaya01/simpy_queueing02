import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from scipy import stats
from collections import defaultdict

import simpy
import random
import pandas as pd
import numpy as np

def compute_avg_queue_length(lengths, times):
    area = 0
    for i in range(1, len(times)):
        duration = times[i] - times[i - 1]
        avg_height = (lengths[i] + lengths[i - 1]) / 2
        area += duration * avg_height
    total_time = times[-1] - times[0] if times else 0
    return area / total_time if total_time > 0 else 0

def run_single_replication(config, rep_id=1):
    env = simpy.Environment()
    NUM_SERVERS = config["NUM_SERVERS"]
    server = simpy.PriorityResource(env, capacity=NUM_SERVERS)
    arrival_times = []
    entity_labels = []
    entity_logs = []
    queue_lengths = []
    queue_times = []
    event_log = []

    # Utilization and breakdown tracking
    total_service_time = 0.0
    server_downtime = [0.0 for _ in range(NUM_SERVERS)]
    server_down_start = [None for _ in range(NUM_SERVERS)]
    server_available = [True for _ in range(NUM_SERVERS)]
    # Entity types
    entity_types_config = config["ENTITY_TYPES"]
    MIN_PRIORITY = config["MIN_PRIORITY"]
    AGING_INTERVAL = config["AGING_INTERVAL"]
    MAX_AGING_STEPS = config["MAX_AGING_STEPS"]
    RENEGE_TIME = config["RENEGE_TIME"]
    BATCH_MIN = config.get("BATCH_MIN", 1)
    BATCH_MAX = config.get("BATCH_MAX", 1)
    ENABLE_BATCH = config.get("ENABLE_BATCH", False)
    SIM_TIME = config["SIM_TIME"]
    WARMUP_TIME = config["WARMUP_TIME"]
    QUEUE_MONITOR_INTERVAL = config["QUEUE_MONITOR_INTERVAL"]
    FAILURE_ENABLED = config.get("FAILURE_ENABLED", False)
    MTBF = config.get("MTBF", 99999)
    MTTR = config.get("MTTR", 1)

    def log_event(time, event_type, entity_id=None, entity_type=None, server_id=None,
                  priority=None, batch_size=None, additional_info=None):
        entry = {
            "Replication": rep_id,
            "Time": time,
            "EventType": event_type,
            "EntityID": entity_id,
            "EntityType": entity_type,
            "ServerID": server_id,
            "Priority": priority,
            "BatchSize": batch_size,
            "QueueLength": len(server.queue),
            "SystemCount": len(server.queue) + server.count,
            "AdditionalInfo": additional_info or ""
        }
        event_log.append(entry)

    def choose_entity_type():
        r = random.random()
        cumulative = 0
        for t in entity_types_config:
            cumulative += t["prob"]
            if r <= cumulative:
                return t
        return entity_types_config[-1]

    def get_current_interarrival_time(current_time):
        p1, p2, p3 = config["PERIOD1"], config["PERIOD2"], config["PERIOD3"]
        m_rate, p_rate, l_rate = config["MORNING_RATE"], config["PEAK_RATE"], config["LATE_RATE"]
        if current_time < p1:
            return m_rate
        elif current_time < p1 + p2:
            return p_rate
        else:
            return l_rate

    def get_next_available_server():
        for i in range(NUM_SERVERS):
            if server_available[i]:
                return i
        return None

    def entity(env, name):
        nonlocal total_service_time
        ent_type = choose_entity_type()
        arrival = env.now
        label = ent_type["name"]
        current_priority = ent_type["priority"]
        wait_start = arrival
        aging_steps = 0

        log_entry = {
            "Replication": rep_id,
            "EntityID": name,
            "Type": label,
            "ArrivalTime": arrival,
            "UsedAging": False,
            "Reneged": False,
            "AgingSteps": 0,
            "AssignedServer": None
        }

        log_event(env.now, "Arrival", entity_id=name, entity_type=label, priority=current_priority)

        while True:
            assigned_server = None
            if FAILURE_ENABLED:
                # Wait for at least one server to be available
                while True:
                    idx = get_next_available_server()
                    if idx is not None:
                        assigned_server = idx
                        break
                    yield env.timeout(0.01)
            with server.request(priority=current_priority) as req:
                results = yield req | env.timeout(RENEGE_TIME) | env.timeout(AGING_INTERVAL)
                now = env.now
                if req in results:
                    if aging_steps > 0:
                        log_entry["UsedAging"] = True
                        log_entry["AgingSteps"] = aging_steps
                    start = now
                    service_time = random.expovariate(1.0 / ent_type["service_time"])
                    log_entry["AssignedServer"] = assigned_server
                    log_event(start, "StartService", entity_id=name, entity_type=label,
                              server_id=assigned_server, priority=current_priority)
                    yield env.timeout(service_time)
                    end = env.now
                    total_service_time += (end - start)
                    log_entry["StartService"] = start
                    log_entry["EndService"] = end
                    log_event(end, "EndService", entity_id=name, entity_type=label,
                              server_id=assigned_server, priority=current_priority)
                    break
                elif now - wait_start >= RENEGE_TIME:
                    log_entry["Reneged"] = True
                    log_event(now, "Reneged", entity_id=name, entity_type=label, priority=current_priority)
                    break
                elif current_priority > MIN_PRIORITY and aging_steps < MAX_AGING_STEPS:
                    current_priority -= 1
                    aging_steps += 1
                    wait_start = now
        entity_logs.append(log_entry)

    def entity_generator(env):
        i = 0
        while env.now < SIM_TIME:
            interarrival = get_current_interarrival_time(env.now)
            yield env.timeout(random.expovariate(1.0 / interarrival))
            batch_size = random.randint(BATCH_MIN, BATCH_MAX) if ENABLE_BATCH else 1
            batch_ids = []
            for _ in range(batch_size):
                i += 1
                name = f"Entity{i}"
                arrival_times.append(env.now)
                label = choose_entity_type()["name"]
                entity_labels.append(label)
                batch_ids.append(name)
                env.process(entity(env, name))
            if batch_size > 1:
                log_event(env.now, "BatchArrival", batch_size=batch_size,
                          additional_info=f"Entities: {batch_ids}")

    def monitor_queue(env):
        while True:
            yield env.timeout(QUEUE_MONITOR_INTERVAL)
            if env.now >= WARMUP_TIME:
                queue_lengths.append(len(server.queue))
                queue_times.append(env.now)

    def server_failure_process(env, idx):
        while True:
            yield env.timeout(random.expovariate(1.0 / MTBF))
            server_down_start[idx] = env.now
            server_available[idx] = False
            log_event(env.now, "ServerBreakdown", server_id=idx)
            yield env.timeout(MTTR)
            if server_down_start[idx] is not None:
                server_downtime[idx] += env.now - server_down_start[idx]
                server_down_start[idx] = None
            server_available[idx] = True
            log_event(env.now, "ServerRepair", server_id=idx)

    env.process(entity_generator(env))
    env.process(monitor_queue(env))
    if FAILURE_ENABLED:
        for i in range(NUM_SERVERS):
            env.process(server_failure_process(env, i))

    env.run(until=SIM_TIME)

    df_log = pd.DataFrame(entity_logs)

    # Utilization calculations
    scheduled_time = NUM_SERVERS * (SIM_TIME - WARMUP_TIME if WARMUP_TIME > 0 else SIM_TIME)
    total_down = sum(server_downtime)
    actual_available_time = scheduled_time - total_down
    scheduled_util = total_service_time / scheduled_time if scheduled_time > 0 else np.nan
    real_util = total_service_time / actual_available_time if actual_available_time > 0 else np.nan

    result = {"Replication": rep_id}
    result["AvgQueueLength"] = compute_avg_queue_length(queue_lengths, queue_times)
    result["AgedEntities"] = df_log["UsedAging"].sum() if "UsedAging" in df_log else 0
    result["Reneged"] = df_log["Reneged"].sum() if "Reneged" in df_log else 0
    result["TotalEntities"] = len(df_log)
    result["ServerUtilization_Scheduled"] = scheduled_util
    result["ServerUtilization_Real"] = real_util
    result["TotalServerDowntime"] = total_down

    for label in df_log["Type"].unique():
        df_served = df_log[(df_log["Type"] == label) & (~df_log["Reneged"])]
        if not df_served.empty:
            result[f"{label}_AvgWait"] = (df_served["StartService"] - df_served["ArrivalTime"]).mean()
            result[f"{label}_AvgService"] = (df_served["EndService"] - df_served["StartService"]).mean()
            result[f"{label}_AvgTotal"] = (df_served["EndService"] - df_served["ArrivalTime"]).mean()
            result[f"{label}_Count"] = len(df_served)
        else:
            result[f"{label}_AvgWait"] = np.nan
            result[f"{label}_AvgService"] = np.nan
            result[f"{label}_AvgTotal"] = np.nan
            result[f"{label}_Count"] = 0

    return result, df_log, pd.DataFrame(event_log), queue_times, queue_lengths, {
        "scheduled_utilization": scheduled_util,
        "real_utilization": real_util,
        "total_down_time": total_down
    }

def run_simulation_multi(config):
    all_summaries = []
    all_entity_logs = []
    all_event_logs = []
    all_queue_times = []
    all_queue_lengths = []
    all_utilizations = []
    for rep_id in range(1, config["REPLICATIONS"] + 1):
        result, df_logs, df_event_log, queue_times, queue_lengths, utilization_stats = run_single_replication(config, rep_id)
        all_summaries.append(result)
        all_entity_logs.append(df_logs)
        all_event_logs.append(df_event_log)
        all_queue_times.append(queue_times)
        all_queue_lengths.append(queue_lengths)
        all_utilizations.append(utilization_stats)
    return all_summaries, all_entity_logs, all_event_logs, all_queue_times, all_queue_lengths, all_utilizations

if __name__ == "__main__":
    # Example config for testing
    config = {
        "SIM_TIME": 100,
        "WARMUP_TIME": 10,
        "REPLICATIONS": 2,
        "NUM_SERVERS": 2,
        "QUEUE_MONITOR_INTERVAL": 0.5,
        "ENTITY_TYPES": [
            {"name": "Type1", "priority": 0, "service_time": 6.0, "prob": 0.5},
            {"name": "Type2", "priority": 1, "service_time": 9.0, "prob": 0.5},
        ],
        "PERIOD1": 33,
        "PERIOD2": 33,
        "PERIOD3": 34,
        "MORNING_RATE": 5,
        "PEAK_RATE": 3,
        "LATE_RATE": 5,
        "ENABLE_BATCH": True,
        "BATCH_MIN": 1,
        "BATCH_MAX": 2,
        "FAILURE_ENABLED": True,
        "MTBF": 60.0,
        "MTTR": 10.0,
        "AGING_INTERVAL": 5,
        "MAX_AGING_STEPS": 2,
        "MIN_PRIORITY": 0,
        "RENEGE_TIME": 12,
    }
    result = run_simulation_multi(config)
    print(result)