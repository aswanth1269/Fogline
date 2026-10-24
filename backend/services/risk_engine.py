def calculate_risk(s,results):
    score=min(s['failed_logins']/12,1)*25+min(s['unique_ports']/40,1)*20+min(s['packet_rate']/600,1)*20+min(s['syn_ratio']/.8,1)*20+(sum(r['attack_probability'] for r in results)/len(results))*15
    score=max(0,min(100,score)); level='High' if score>=75 else 'Medium' if score>=45 else 'Low'; return round(score,1),level
