# -*- coding: utf-8 -*-
"""主程序：串联四个子问题，汇总 figures/all_results.json。
复用已验证的逐问结果(figures/problem_*_results.json)，支持从空缓存完整运行。"""
import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import utils
import problem1, problem2, problem3, problem4

P2_TIME_LIMIT = int(os.environ.get('P2_TIME_LIMIT', '300'))


def _load_or_run(name, path, runner, validator):
    """已有有效结果则复用，否则求解。"""
    fp = utils.FIGDIR / path
    if fp.exists() and fp.stat().st_size > 100:
        res = json.loads(fp.read_text(encoding='utf-8'))
        try:
            validator(res)
            print(f"[main] {name} 复用已验证结果 {path}")
            return res
        except Exception as e:
            print(f"[main] {name} 缓存校验失败({e})，重算")
    return runner()


def main():
    print("=== 山区洪涝无人机运输通信协同优化 — 四问求解 ===", flush=True)
    r1 = _load_or_run('P1', 'problem_1_results.json', problem1.run, problem1.validate_capability)
    problem1.validate_capability(r1)

    r2 = _load_or_run('P2', 'problem_2_results.json',
                      lambda: problem2.run(time_limit=P2_TIME_LIMIT), problem2.validate_capability)
    problem2.validate_capability(r2)

    r3 = _load_or_run('P3', 'problem_3_results.json', problem3.run, problem3.validate_capability)
    problem3.validate_capability(r3)

    r4 = _load_or_run('P4', 'problem_4_results.json', problem4.run, problem4.validate_capability)
    problem4.validate_capability(r4)

    all_results = {
        'problem1': {'safe_payload_matrix': r1['safe_payload_matrix'],
                     'summary': r1['summary'], 'sensitivity_rho': r1['sensitivity_rho']},
        'problem2': {'objective': r2['objective'], 'n_trips': len(r2['trips']),
                     'resource_audit': r2['resource_audit']},
        'problem3': {'comm_diagnosis': r3['comm_diagnosis'],
                     'joint_objective': r3['joint_objective']},
        'problem4': {'supernodes': r4['supernodes'], 'inventory': r4['inventory'],
                     'partitions': {k: {'total_config': v['total_config'],
                                        'total_gap': v['total_gap'],
                                        'gap_by_type': v['gap_by_type']}
                                    for k, v in r4['partitions'].items()}},
    }
    utils.save_json(utils.FIGDIR / 'all_results.json', all_results)
    print("\n=== 全部四问求解完成，all_results.json 已保存 ===", flush=True)
    return all_results


if __name__ == '__main__':
    main()
