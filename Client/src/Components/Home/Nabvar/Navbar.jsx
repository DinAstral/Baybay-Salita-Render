import { useEffect, useState, useContext } from "react";
import { Link } from "react-router-dom";
import { Link as LinkScroll } from "react-scroll";
import "./Navbar.css";
import logo from "../../../assets/LogoFinal.png";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faBars } from "@fortawesome/free-solid-svg-icons";
import { UserContext } from "../../../../context/userContext"; // Adjust the import path as needed

function Navbar() {
  const [sticky, setSticky] = useState(false);
  const [mobileMenu, setMobileMenu] = useState(false);
  const { user, clearCookie } = useContext(UserContext); // Access the user context

  useEffect(() => {
    const handleScroll = () => {
      window.scrollY > 400 ? setSticky(true) : setSticky(false);
    };
    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const toggleMenu = () => {
    setMobileMenu(!mobileMenu);
  };

  return (
    <nav className={`container-nav ${sticky ? "dark-nav" : ""}`}>
      <img src={logo} alt="" className="logo" draggable="false" />
      <ul className={mobileMenu ? "" : "hide-mobile-menu"}>
        <li>
          <LinkScroll
            to="hero"
            smooth={true}
            offset={-100}
            duration={100}
            className="cursor-pointer"
          >
            Home
          </LinkScroll>
        </li>
        <li>
          <LinkScroll
            to="services"
            smooth={true}
            offset={-290}
            duration={100}
            className="cursor-pointer"
          >
            Services
          </LinkScroll>
        </li>
        <li>
          <LinkScroll
            to="about"
            smooth={true}
            offset={-150}
            duration={100}
            className="cursor-pointer"
          >
            About
          </LinkScroll>
        </li>
        <li>
          <LinkScroll
            to="contact"
            smooth={true}
            offset={-300}
            duration={100}
            className="cursor-pointer"
          >
            Contact
          </LinkScroll>
        </li>
        <li>Download</li>
        {user && (
          <>
            <li>
              <Link
                to={`/${user.role.toLowerCase()}Dashboard`}
                className="btn-home"
              >
                Dashboard
              </Link>
            </li>
            <li>
              <button onClick={() => clearCookie()} className="btn-home">
                Logout
              </button>
            </li>
          </>
        )}
        {!user && (
          <li>
            <Link to="/login" className="btn-home">
              Login
            </Link>
          </li>
        )}
      </ul>
      <FontAwesomeIcon
        icon={faBars}
        size="2x"
        className="menu-icon"
        onClick={toggleMenu}
      />
    </nav>
  );
}

export default Navbar;
