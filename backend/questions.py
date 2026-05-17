# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Scenario Questions Database
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_POOL = {
    "Computer Science": [
        {
            "id": "cs_vault",
            "next_question": "Imagine you are designing a high-tech smart lock system for a secure vault. What is the most satisfying part of the setup?",
            "options": [
                {"text": "💻 Writing the software algorithm that encrypts credentials and handles face recognition", "value": "I love coding software algorithms, writing secure programs, and designing encryption codes."},
                {"text": "🔌 Wiring up the electronic locks, batteries, microchips, and fingerprint sensors", "value": "I want to wire electronic microchips, sensors, breadboards, and hardware controllers."},
                {"text": "🛠️ Assembling the physical steel frame, hydraulic locking bars, and mechanical hinges", "value": "Tinkering with heavy mechanical locks, physical structural gearboxes, and mechanical joints."},
                {"text": "📊 Sourcing parts under budget and selling the high-end security vault to banks", "value": "Negotiating with suppliers, pitching startup business ideas, and planning budgets."}
            ]
        },
        {
            "id": "cs_game_lag",
            "next_question": "A gaming startup is launching a multiplayer game, but the server is experiencing high lag. What task would you tackle first?",
            "options": [
                {"text": "💻 Optimizing database queries, server scripts, and networking sockets", "value": "I enjoy coding backend servers, automating systems, writing databases, and networking scripts."},
                {"text": "📈 Redesigning the pricing model, cosmetic store, and in-game transaction fees", "value": "I want to design corporate pricing strategies, manage finance budgets, and track marketing sales."},
                {"text": "⚙️ Optimizing the game's structural physics engine and mechanical collision loops", "value": "Analyzing speed forces, kinematics physics, mechanical motion, and mechanics."},
                {"text": "🔌 Checking if local network router hardware, microchips, and cable signals are stable", "value": "Testing electrical wiring connections, hardware components, and circuit microcontrollers."}
            ]
        }
    ],
    "Mechanical Engineering": [
        {
            "id": "mech_kart",
            "next_question": "Your college friends are building a miniature electric racing go-kart. Which role do you claim first?",
            "options": [
                {"text": "🛠️ Designing the physical metal chassis, steering linkages, and shock suspension", "value": "Designing mechanical CAD models, gear mechanisms, automobile chassis, and manufacturing engines."},
                {"text": "🔌 Connecting the battery cells and soldering the electronic speed controllers", "value": "Wiring electrical circuits, connecting power systems, battery cells, and soldering wires."},
                {"text": "💻 Coding the self-driving steering logic and sensor feedback loops", "value": "Writing automation python code, scripting software programs, and coding app databases."},
                {"text": "📢 Managing the team budget, seeking sponsors, and preparing a pitching deck", "value": "Organizing project budgets, pitching business ideas, and managing startup finances."}
            ]
        },
        {
            "id": "mech_robot",
            "next_question": "A factory is automating its assembly line with advanced robotic arms. What part of the robot is most fascinating?",
            "options": [
                {"text": "🛠️ The mechanical joints, gears, fluid pumps, and high-torque motors", "value": "Tinkering with gears, mechanical levers, hydraulic valves, automotive motors, and machine assembly."},
                {"text": "🔌 The microchip circuit boards, copper wiring, and optical feedback sensors", "value": "Designing printed circuit boards, choosing resistors, soldering wires, and electrical components."},
                {"text": "💻 The software script that coordinates multiple arms using loops and trigonometry", "value": "Coding logical control algorithms, programming software automation scripts, and databases."},
                {"text": "📊 Calculating factory ROI, labor cost-savings, and pitching to corporate clients", "value": "Managing team operations, calculating profit margins, HR coordinating, and financial business reports."}
            ]
        }
    ],
    "Electrical Engineering": [
        {
            "id": "elec_garden",
            "next_question": "You are building an automated watering system for your terrace garden. What do you enjoy setting up most?",
            "options": [
                {"text": "🔌 Soldering the breadboard, wiring moisture sensors, and connecting LEDs", "value": "Wiring electronic microchips, sensors, breadboards, voltage currents, and electrical grids."},
                {"text": "🏗️ Assembling the physical plastic piping, pressure control valves, and stands", "value": "Building concrete foundations, laying architectural layouts, structural design, and water piping structures."},
                {"text": "💻 Writing the software timer code that triggers water pumps based on sensor thresholds", "value": "Coding automation software scripts, writing database systems, and web programming."},
                {"text": "🤝 Negotiating fruit seeds and soil costs from local nurseries in bulk", "value": "Negotiating bulk deals with suppliers, tracking business budgets, and finance margins."}
            ]
        },
        {
            "id": "elec_lamp",
            "next_question": "You want to design a custom smart desk lamp that adjusts color based on temperature. What excites you?",
            "options": [
                {"text": "🔌 Designing the custom printed circuit board (PCB) and choosing resistors", "value": "Soldering microchips, wiring battery terminals, microcontrollers, and electrical circuit boards."},
                {"text": "🛠️ Hand-crafting the modern lamp stand from polished wood and brass links", "value": "Designing physical CAD products, metal machining, mechanical joints, and assembly."},
                {"text": "💻 Coding the smartphone app that controls the lamp color using Bluetooth", "value": "Writing mobile app codes, fullstack web development, and cloud software engineering."},
                {"text": "📢 Researching customer demand, pricing the lamp, and launching a startup launch", "value": "Launching e-commerce startups, marketing campaigns, branding design, and corporate sales."}
            ]
        }
    ],
    "Civil Engineering": [
        {
            "id": "civil_bridge",
            "next_question": "A damaged highway bridge in your city needs urgent reconstruction. What is your priority task?",
            "options": [
                {"text": "🏗️ Calculating concrete load distributions, foundation depths, and structural safety", "value": "Calculating civil structural loads, surveying soils, building concrete foundations, and bridge blueprints."},
                {"text": "🧪 Testing the river water's chemistry and biological toxicity near construction", "value": "Testing chemical solutions, pH balances, biology cell microbes, and ecological water analysis."},
                {"text": "💻 Designing a database system to track material fleets using GPS coordination", "value": "Developing software databases, writing backend code, and tracking automation systems."},
                {"text": "📊 Managing the public tender budget, labor union contracts, and schedules", "value": "Managing project timelines, HR coordination, business contracting, and finance budgets."}
            ]
        },
        {
            "id": "civil_model",
            "next_question": "You are building a miniature model of a sustainable smart eco-city. Where is your attention?",
            "options": [
                {"text": "🏗️ Laying structural roads, building concrete bases, and carving tunnels", "value": "Drafting structural layouts, casting concrete, bridge designs, and civil building materials."},
                {"text": "🔌 Wiring miniature smart streetlights with micro solar panel circuits", "value": "Soldering micro solar panel circuits, wiring LED lamps, and battery currents."},
                {"text": "💻 Coding an interactive dashboard that simulates utility flows in Python", "value": "Programming utility simulation algorithms, website UI dashboards, and software databases."},
                {"text": "📈 Planning residential property zones, business tax systems, and retail marketing", "value": "Planning finance investments, zoning regulations, economy research, and retail business."}
            ]
        }
    ],
    "Biology": [
        {
            "id": "bio_purple_leaves",
            "next_question": "A strange plant infection is turning the school garden leaves purple and wilting them. What do you do?",
            "options": [
                {"text": "🔬 Preparing a leaf slide to analyze cellular structures and plant DNA under a microscope", "value": "Analyzing biological cells, gene structures, plant DNA, botany, and medical microscope slides."},
                {"text": "🧪 Testing leaf extract reactions with acidic and basic chemical solutions in a lab", "value": "Testing chemistry solutions, molecular chemical reactions, titration, and beaker chemical formulas."},
                {"text": "📐 Measuring the exact speed, light wave angle, and physics forces spreading the wilt", "value": "Analyzing force dynamics, wave light refraction, lenses, and physics lab experiments."},
                {"text": "📢 Writing a commercial proposal for a local organic fertilizer brand to cure it", "value": "Writing business plans, pitching products, marketing campaigns, and startup partnerships."}
            ]
        },
        {
            "id": "bio_health_camp",
            "next_question": "Your neighborhood is holding a voluntary health checkup camp. What role matches your interest?",
            "options": [
                {"text": "🩺 Checking biological symptoms, explaining heart mechanics, and doctor consultations", "value": "Studying human anatomy, cellular biological health, medical symptoms, and clinic care."},
                {"text": "🧪 Titrating chemical indicators to check glucose and pH levels in samples", "value": "Analyzing chemical properties, blood chemistry balance, reagents, and laboratory mixtures."},
                {"text": "🔌 Calibrating electronic heart rate sensors and battery voltage monitors", "value": "Wiring electronic sensors, calibrating circuit chips, and electrical breadboard signals."},
                {"text": "📋 Coordinating volunteer registers, patient queue flows, and donor finances", "value": "Coordinating volunteer teams, managing registrations, HR planning, and finance cash box."}
            ]
        }
    ],
    "Chemistry": [
        {
            "id": "chem_serum",
            "next_question": "A cosmetic startup is developing an organic herbal skin serum. Which role sounds most exciting?",
            "options": [
                {"text": "🧪 Formulating the chemical emulsion, mixing active ingredients, and balancing pH levels", "value": "Formulating chemical molecular compounds, mixing chemical solutions, pH testing, and lab reactions."},
                {"text": "🔬 Researching cell tissue absorption and testing herbal extract biology under a microscope", "value": "Analyzing skin cells biology, microscope tissue slides, biochemistry, and plant genetics."},
                {"text": "🛠️ Assembling the mechanical rotor valves and automatic packaging machines in the factory", "value": "Machining mechanical rotor pumps, assembly line engines, and automated gears."},
                {"text": "📢 Designing the brand identity, pricing margins, and running social media advertisements", "value": "Creating marketing sales campaigns, setting startup pricing margins, and branding strategy."}
            ]
        },
        {
            "id": "chem_lake",
            "next_question": "During a conservation drive at a highly polluted lake, which activity captures your focus?",
            "options": [
                {"text": "🧪 Mixing chemical indicators with lake water to test for toxic heavy metal residues", "value": "Mixing chemical reagents, testing compound formulas, acid base titrations, and chemical safety."},
                {"text": "🌿 Studying lake algae growth rates, aquatic cellular organisms, and microbes", "value": "Studying aquatic biology ecosystems, microbes, lake botany, and zoology organisms."},
                {"text": "📐 Calculating water flow rates, physics currents, and designing charcoal filtration beds", "value": "Analyzing physics fluid mechanics, force currents, sand filtration structures, and energy."},
                {"text": "🤝 Pitching findings to local corporate sponsors to secure cleanup funds", "value": "Pitching to corporate sponsors, negotiating contract terms, and organizing finance investments."}
            ]
        }
    ],
    "Physics": [
        {
            "id": "phys_workshop",
            "next_question": "Your school science club offers four specialized weekend tracks. Which one do you enroll in?",
            "options": [
                {"text": "🌌 Rocket Propulsion & Space: Exploring orbital gravity, forces, and telescope lenses", "value": "Studying astrophysics space gravity, light optics refraction, laser physics, and kinetic forces."},
                {"text": "🧬 Genetic Editing: Learning about DNA extraction and cellular biology laboratory work", "value": "Extracting plant DNA, cell genetics research, microbiology, and biological mutations."},
                {"text": "💻 Game Programming: Writing code in Python to compile structural logic models", "value": "Coding algorithms, software application compiling, web development, and cloud databases."},
                {"text": "📈 Young Tycoons: Pitching startup ideas, learning financing, and stock market budgets", "value": "Pitching corporate strategies, stocks, profit budgeting, and marketing businesses."}
            ]
        },
        {
            "id": "phys_cooker",
            "next_question": "You are designing an eco-friendly parabolic solar cooker. What satisfies your problem-solving instinct?",
            "options": [
                {"text": "📐 Calculating focus light angles, parabolic curvature forces, and heat absorption waves", "value": "Calculating light refraction focal angles, solar heat radiation waves, gravity, and kinetic physics."},
                {"text": "🧪 Analyzing the molecular chemical coatings that retain thermal heat inside the pot", "value": "Formulating chemical molecular compounds, polymer coatings, and thermal chemical reactions."},
                {"text": "💻 Coding an automated Arduino script that rotates mirrors with solar GPS tracking", "value": "Coding Arduino sensor scripts, programming software coordinates, and database logging."},
                {"text": "📊 Drafting the budget sheet, sourcing cheap aluminum sheets, and shipping systems", "value": "Managing budget finance margins, organizing distribution operations, and corporate sourcing."}
            ]
        }
    ],
    "Management": [
        {
            "id": "mgmt_juice",
            "next_question": "Your classmates are setting up a refreshment juice stall during the annual athletic meet. What is your role?",
            "options": [
                {"text": "📊 Managing the price list, sourcing fruits in bulk, and managing the cash logs", "value": "Managing budget pricing, negotiating bulk fruit sourcing, finance operations, and startup profit margins."},
                {"text": "🛠️ Assembling a manual hand-press squeezer with levers and physical gears for fast juicing", "value": "Tinkering with physical gears, manual leverage mechanics, mechanical squeezer, and engines."},
                {"text": "🧪 Analyzing fruit sugar concentrations, acidity, and chemical preservative balances", "value": "Analyzing food chemistry sugars, pH balance, organic molecules, and compound formulas."},
                {"text": "💻 Setting up digital QR payment scanner and writing a spreadsheet code to log sales", "value": "Programming database spreadsheets, coding sales trackers, and setting digital API payment systems."}
            ]
        },
        {
            "id": "mgmt_recycled",
            "next_question": "A local brand wants to launch a sustainable, recycled t-shirt line. What role fits you best?",
            "options": [
                {"text": "📢 Launching marketing ads, recruiting models, and setting sales targets", "value": "Running branding marketing campaigns, recruiting staff, organizing teams, sales pitches, and BBA MBA management."},
                {"text": "🧪 Testing non-toxic chemical dyes to ensure they don't wash off in water", "value": "Testing chemical compounds, molecular dye formulas, polymer textiles, and chemistry labs."},
                {"text": "🛠️ Programming the automated weaving looms and mechanical spinning pulleys", "value": "Assembling mechanical weaving gears, automatic textile engines, and industrial machines."},
                {"text": "💻 Building the e-commerce Shopify website and database server portals", "value": "Building software databases, website coding, e-commerce backend code, and server setups."}
            ]
        }
    ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Starter Profiling & Bridging Questions (Rule-based to establish initial streams)
# ─────────────────────────────────────────────────────────────────────────────

SEED_QUESTIONS = {
    "education": {
        "text": "Hello! I am your University Admissions Career Counsellor. Let's find your ideal academic career path together! First, what is your current education level?",
        "options": [
            {"text": "🎓 12th Standard Completed", "value": "12th Standard"},
            {"text": "🏫 10th Standard Completed", "value": "10th Standard"},
            {"text": "✨ Something else / Other", "value": "Other"}
        ]
    },
    "stream_12th": {
        "text": "Excellent! Which academic stream did you study in your 12th Standard?",
        "options": [
            {"text": "📐 Science with Maths (PCM)", "value": "PCM"},
            {"text": "🧬 Science with Biology (PCB)", "value": "PCB"},
            {"text": "💼 General / Commerce Stream", "value": "General / Commerce"},
            {"text": "🌟 Something else / Other stream", "value": "Other"}
        ]
    },
    "bridge_pcm": {
        "text": "Great, PCM stream! Let's explore your strengths in Engineering, Sciences, or Tech with a few quick real-life scenarios. Simply select the option that feels most like you, or write your own custom response!"
    },
    "bridge_pcb": {
        "text": "Wonderful, PCB stream! Let's explore your interest across Medical Sciences, Biotech, or Chemistry with a few real-life scenarios. Choose the options that fit best or tell me in your own words!"
    },
    "bridge_general": {
        "text": "Excellent! Let's discover your alignment across Business Management, Technology, or Pure Sciences with a few situations. Select your natural choice or feel free to type your own response!"
    },
    "general_start": {
        "id": "general_start_question",
        "next_question": "Imagine your school or colony is organizing a massive annual cultural and science fest. What major role would you claim first?",
        "options": [
            {"text": "📅 The Organizer: Coordinating volunteer teams, budgeting costs, and managing sponsors", "value": "I enjoy organizing event budgets, pitching ideas to sponsors, managing volunteers, and business operations."},
            {"text": "💻 The Tech Lead: Building the registration website, databases, and digital banners", "value": "I love coding websites, designing app databases, writing software algorithms, and system automation."},
            {"text": "🔬 The Lab Curator: Running live chemical reactions and biological cell microscope displays", "value": "Running lab chemistry experiments, molecular biology cells, DNA testing, and botany micro-organism studies."},
            {"text": "🛠️ The Design Engineer: Assembling the structural entry arches and mechanical robotic projects", "value": "Tinkering with mechanical gears, motors, structural bridge layouts, concrete bases, and electrical circuits."}
        ]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Question Picker Logic
# ─────────────────────────────────────────────────────────────────────────────

def get_adaptive_question(current_predictions: dict, asked_question_ids: list) -> dict:
    """
    Looks at the current top prediction categories (from our ML model),
    finds an unasked question in that category, and returns it.
    If no questions are left in the top category, it goes to the second-highest,
    and finally falls back to any unasked question in the entire pool.
    """
    # Exclude categories we don't have pools for directly (e.g. Map specific streams)
    # The keys in QUESTION_POOL are the primary headings
    # We rank the predictions from ML model
    for category in current_predictions.keys():
        # Match standard prediction string to QUESTION_POOL keys
        pool_key = None
        if "Computer Science" in category:
            pool_key = "Computer Science"
        elif "Mechanical" in category:
            pool_key = "Mechanical Engineering"
        elif "Electrical" in category:
            pool_key = "Electrical Engineering"
        elif "Civil" in category:
            pool_key = "Civil Engineering"
        elif "Biology" in category:
            pool_key = "Biology"
        elif "Chemistry" in category:
            pool_key = "Chemistry"
        elif "Physics" in category:
            pool_key = "Physics"
        elif "Management" in category:
            pool_key = "Management"
            
        if pool_key and pool_key in QUESTION_POOL:
            questions = QUESTION_POOL[pool_key]
            for q in questions:
                if q["id"] not in asked_question_ids:
                    return q
                    
    # Fallback to absolute list of any unasked question
    for pool_key, questions in QUESTION_POOL.items():
        for q in questions:
            if q["id"] not in asked_question_ids:
                return q
                
    # If all asked, return a default final question (should not happen in 4 turns)
    return QUESTION_POOL["Computer Science"][0]
